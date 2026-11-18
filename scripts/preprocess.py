"""Build one sensor graph per IED from Brainstorm epoch and channel files."""

import argparse
import re
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import torch
from torch_geometric.data import Data


FEATURE_NAMES = ["t0_value", "t0_sign", "neighbor_abs_mean",
                 "temporal_consistency", "spatial_gradient"]
EPS = 1e-8


def meg_channels(path):
    channels = np.atleast_1d(loadmat(path, squeeze_me=True, struct_as_record=False)["chan"].Channel)
    indices = [i for i, channel in enumerate(channels) if channel.Type == "MEG"]
    coords = [np.asarray(channels[i].Loc).reshape(-1)[:3] for i in indices]
    return np.asarray(indices), np.asarray(coords, dtype=float)


def sensor_adjacency(coords):
    """Reproduce the helmet triangulation used in the DFF project."""
    center = coords.mean(axis=0)
    radius = np.linalg.norm(coords - center, axis=1).mean()

    def sphere_error(params):
        radii = np.linalg.norm(coords - params[:3], axis=1)
        return np.abs(radii - params[3]).mean()

    fit = minimize(sphere_error, np.r_[center, radius], method="Nelder-Mead",
                   options={"maxiter": 10000, "xatol": 1e-12, "fatol": 1e-12})
    unit = coords - fit.x[:3]
    unit /= np.linalg.norm(unit, axis=1, keepdims=True)
    faces = ConvexHull(unit / np.linalg.norm(unit, axis=1, keepdims=True)).simplices

    x, y, z = coords.T
    azimuth = np.arctan2(y, x)
    elevation = -np.arctan2(z, np.hypot(x, y))
    radius_2d = 2 * np.cos((np.pi / 2 - elevation) / 2) / np.sqrt(2)
    cap = np.column_stack((radius_2d * np.cos(azimuth), radius_2d * np.sin(azimuth)))
    border = ConvexHull(cap).vertices
    interior = ~np.isin(faces, border).all(axis=1)
    a, b, c = coords[faces].transpose(1, 0, 2)
    perimeter = (np.linalg.norm(a - b, axis=1) + np.linalg.norm(a - c, axis=1)
                 + np.linalg.norm(b - c, axis=1))
    limit = perimeter[interior].mean() + 6 * perimeter[interior].std(ddof=1)
    kept = interior & (perimeter <= limit)
    # Restore the shortest incident face for each sensor omitted at the cap edge.
    for node in np.unique(faces):
        if node not in faces[kept]:
            incident = np.flatnonzero((faces == node).any(axis=1))
            kept[incident[np.argmin(perimeter[incident])]] = True
    tri = faces[kept]
    edges = np.vstack((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]))
    i, j = np.unique(np.sort(edges, axis=1), axis=0).T
    distance = np.linalg.norm(coords[i] - coords[j], axis=1)
    adj = np.zeros((len(coords), len(coords)), dtype=np.float32)
    adj[i, j] = adj[j, i] = distance / distance.max()
    return adj


def epoch_features(epoch, meg, adj, coords):
    time = np.asarray(epoch.Time).ravel()
    peak = np.argmin(np.abs(time))
    if abs(time[peak]) > 1e-12 or peak < 2 or peak + 2 >= len(time):
        raise ValueError("Each IED epoch must include time zero and two samples on either side.")
    window = np.asarray(epoch.F[meg, peak - 2:peak + 3], dtype=np.float64)
    value = window[:, 2]
    neighbor_mean = np.zeros(len(meg))
    gradient = np.zeros(len(meg))
    for i in range(len(meg)):
        neighbors = np.flatnonzero(adj[i])
        if len(neighbors):
            neighbor_mean[i] = np.abs(value[neighbors]).mean()
            distance = np.linalg.norm(coords[neighbors] - coords[i], axis=1)
            gradient[i] = (np.abs(value[i] - value[neighbors]) / (distance + EPS)).mean()
    consistency = 1 / (1 + window.std(axis=1) / (np.abs(window.mean(axis=1)) + EPS))
    return np.column_stack((value, np.sign(value), neighbor_mean, consistency, gradient)).astype("float32")


def build_graph(root, legacy_flags=False):
    subjects = sorted(p for p in root.glob("sub-*") if p.is_dir())
    if not subjects:
        raise ValueError(f"No subject folders found in {root}")
    events, epoch_fields, graph_subjects = [], [], []
    for subject_id, directory in enumerate(subjects):
        meg, coords = meg_channels(directory / "chan.mat")
        adj = sensor_adjacency(coords)
        row, col = np.nonzero(adj)
        edges = torch.from_numpy(np.stack((row, col))).long()
        weights = torch.from_numpy(adj[row, col, None])
        epochs = loadmat(directory / "data.mat", squeeze_me=True, struct_as_record=False)
        fields = sorted((key for key in epochs if re.fullmatch(r"data_\d+", key)),
                        key=lambda key: int(key.split("_")[1]))
        for epoch_index, field in enumerate(fields):
            epoch = epochs[field]
            flags = np.asarray(epoch.ChannelFlag).ravel()[meg].astype(np.int64)
            if legacy_flags:
                flags = np.select([flags == 3, flags == 2], [2, 1], default=flags)
            if not np.isin(flags, [-1, 0, 1, 2]).all():
                raise ValueError("Expected manuscript ChannelFlag values -1, 0, 1 and 2. "
                                 "Use --legacy-flags for original DFF exports.")
            flag = torch.from_numpy(flags)
            events.append(Data(
                x=torch.from_numpy(epoch_features(epoch, meg, adj, coords)),
                edge_index=edges, edge_attr=weights, channel_flag=flag,
                y=torch.where(flag == 2, 1, torch.where(flag == 0, 0, -1)),
                subject_id=torch.full_like(flag, subject_id), epoch_index=torch.full_like(flag, epoch_index),
                channel_id=torch.arange(len(meg)), graph_id=torch.full_like(flag, len(events)),
            ))
            epoch_fields.append(field)
            graph_subjects.append(directory.name)

    # Only edge indices receive offsets when the event graphs are concatenated.
    offsets = np.cumsum([0] + [event.num_nodes for event in events[:-1]])
    graph = Data(edge_index=torch.cat([e.edge_index + int(n) for e, n in zip(events, offsets)], dim=1))
    for key in ("x", "edge_attr", "y", "channel_flag", "subject_id", "epoch_index", "channel_id", "graph_id"):
        graph[key] = torch.cat([event[key] for event in events])
    graph.subject_names = [p.name for p in subjects]
    graph.graph_subjects = graph_subjects
    graph.graph_epoch_fields = epoch_fields
    graph.feature_names = FEATURE_NAMES
    graph.num_epoch_graphs = len(events)
    graph.nodes_per_epoch = events[0].num_nodes
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subjects", type=Path, help="Folder containing sub-*/chan.mat and data.mat")
    parser.add_argument("output", type=Path, help="Output PyTorch Geometric graph (.pt)")
    parser.add_argument("--legacy-flags", action="store_true", help="Read the original DFF 0/1/2/3 annotations")
    args = parser.parse_args()
    graph = build_graph(args.subjects, args.legacy_flags)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, args.output)
    print(f"{len(graph.subject_names)} subjects, {graph.num_epoch_graphs} events, {graph.num_nodes} nodes")


if __name__ == "__main__":
    main()
