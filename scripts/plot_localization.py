"""Plot expert-referenced dipole localization differences (Figure 8)."""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN


RESULT_DIRS = {
    "Expert": ("event1_dff",),
    "DFF": ("event1_dff_predict",),
    "DFF-noSpatialGradient": ("event1_dff_ablation", "event1_dff_predict_nogradient"),
    "GraphSAGE": ("event1_dff_graphsage", "event1_dff_predict_graphsage"),
}
COLORS = {"DFF": "#CDB4DB", "DFF-noSpatialGradient": "#B8C0FF", "GraphSAGE": "#A2D2FF"}
TRIAL_FILE = re.compile(r"dipoles_event1_dff_trial(\d{3})(?:_(\d+))?\.mat", re.IGNORECASE)


def load_locations(subject_dir, method):
    """Return fitted positions in mm keyed by trial and dipole ordinal."""
    for name in RESULT_DIRS[method]:
        trials = {}
        for path in sorted((subject_dir / name).glob("*.mat")):
            match = TRIAL_FILE.fullmatch(path.name)
            if match:
                trial = int(match[1])
                revision = int(match[2] or 0)
                if trial not in trials or revision >= trials[trial][0]:
                    trials[trial] = revision, path
        if trials:
            break

    locations = {}
    for trial, (_, path) in sorted(trials.items()):
        dipoles = loadmat(path, variable_names=["Dipole"])["Dipole"]
        for ordinal, dipole in enumerate(dipoles.ravel()):
            locations[trial, ordinal] = dipole["Loc"].reshape(3) * 1000.0
    return locations


def cluster_centroids(points):
    labels = DBSCAN(eps=20.0, min_samples=3).fit_predict(points)
    return np.array([
        points[labels == label].mean(axis=0) for label in np.unique(labels) if label >= 0
    ]).reshape(-1, 3)


def localization_distances(root):
    paired_rows = []
    cluster_rows = []
    for subject in sorted(path for path in root.iterdir() if path.is_dir()):
        expert = load_locations(subject, "Expert")
        for method in COLORS:
            predicted = load_locations(subject, method)
            pairs = sorted(expert.keys() & predicted.keys())
            if not pairs:
                continue
            reference = np.array([expert[key] for key in pairs])
            estimate = np.array([predicted[key] for key in pairs])
            distances = np.linalg.norm(estimate - reference, axis=1)
            for (trial, ordinal), distance in zip(pairs, distances):
                paired_rows.append((subject.name, method, trial, ordinal, distance))

            costs = cdist(cluster_centroids(reference), cluster_centroids(estimate))
            rows, cols = linear_sum_assignment(costs)
            for r, c in zip(rows, cols):
                cluster_rows.append((subject.name, method, r, c, costs[r, c]))

    paired = pd.DataFrame(paired_rows, columns=[
        "subject", "method", "trial", "dipole", "distance_mm"
    ])
    clusters = pd.DataFrame(cluster_rows, columns=[
        "subject", "method", "expert_cluster", "predicted_cluster", "distance_mm"
    ])
    return paired, clusters


def plot_distances(paired, clusters):
    methods = [method for method in COLORS if method in set(paired["method"])]
    subjects = sorted(paired["subject"].unique())
    colors = [COLORS[method] for method in methods]
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained",
                                gridspec_kw={"width_ratios": [1, 1, 1.6]})
    for row, frame, label in [
        (0, clusters, "Cluster-centroid distance"),
        (1, paired, "Paired dipole distance"),
    ]:
        mean_ax, std_ax, subject_ax = axes[row]
        summary = frame.groupby("method")["distance_mm"].agg(["mean", "std"]).reindex(methods)
        mean_ax.bar(methods, summary["mean"], yerr=summary["std"], color=colors, capsize=3)
        std_ax.bar(methods, summary["std"], color=colors)
        for ax in (mean_ax, std_ax):
            ax.tick_params(axis="x", labelrotation=28)
        mean_ax.set_title("a" if row == 0 else "b", loc="left", fontweight="bold")
        mean_ax.set_title("Mean ± SD")
        std_ax.set_title("Standard deviation")
        mean_ax.set_ylabel(label + " (mm)")
        std_ax.set_ylabel("SD (mm)")
        per_subject = frame.groupby(["subject", "method"])["distance_mm"].mean()
        x = np.arange(len(subjects))
        width = 0.8 / len(methods)
        for i, method in enumerate(methods):
            means = [per_subject.get((subject, method), np.nan) for subject in subjects]
            subject_ax.bar(x + (i - (len(methods) - 1) / 2) * width, means,
                           width, color=COLORS[method], label=method)
        subject_ax.set_xticks(x, subjects, rotation=28)
        subject_ax.set_ylabel("Mean distance (mm)")
        subject_ax.set_title("Individual subjects")
    axes[0, 2].legend(frameon=False)
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path, help="Brainstorm data directory containing subject folders")
    parser.add_argument("--output", type=Path, help="Save the figure at this path")
    args = parser.parse_args()
    paired, clusters = localization_distances(args.results_root)
    if paired.empty:
        parser.error("No matching expert and model dipoles found under the results directory.")
    for name, frame in [("Paired dipoles", paired), ("Matched cluster centroids", clusters)]:
        print(name)
        print(frame.groupby("method")["distance_mm"].agg(["count", "mean", "std"]).to_string())
    figure = plot_distances(paired, clusters)
    if args.output:
        figure.savefig(args.output, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
