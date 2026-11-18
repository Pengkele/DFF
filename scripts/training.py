"""Subject-level cross-validation for channel classification."""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from model import MODELS


ROOT = Path(__file__).resolve().parents[1]
BASELINES = ("MLP", "GCN", "GIN", "GraphSAGE", "BWGNN")


def load_graph(path):
    graph = torch.load(path, map_location="cpu", weights_only=False)
    flags = graph.channel_flag.reshape(-1)
    if "graph_flag3_count" in graph:
        # The original cache distinguished selected (3) and uncertain (2) dipoles.
        flags = torch.where(flags == 3, 2, torch.where(flags == 2, 1, flags))
        del graph.graph_flag3_count
    graph.channel_flag = flags
    graph.y = torch.where(flags == 2, 1, torch.where(flags == 0, 0, -1))
    return graph


def graph_without_bad_channels(raw):
    keep = raw.channel_flag.reshape(-1) != -1
    edges, attributes = subgraph(keep, raw.edge_index, raw.edge_attr, relabel_nodes=True)
    return Data(
        x=raw.x[keep].float(),
        edge_index=edges,
        edge_attr=attributes.reshape(-1, 1).float(),
        y=raw.y[keep].reshape(-1).long(),
        graph_id=raw.graph_id[keep],
        subject_id=raw.subject_id[keep],
        epoch_index=raw.epoch_index[keep],
        channel_id=raw.channel_id[keep],
        subject_names=raw.subject_names,
    )


def subject_folds(graph):
    """Balance the number of labelled nodes from each class across five folds."""
    subjects = graph.subject_id.unique().tolist()
    if len(subjects) < 5:
        raise ValueError("Five-fold cross-validation requires at least five development subjects.")

    rows = []
    for subject in subjects:
        labels = graph.y[(graph.subject_id == subject) & (graph.y >= 0)]
        counts = torch.bincount(labels, minlength=2).float()
        rows.append((int(counts[1]), int(counts.sum()), subject, counts))

    target = sum(row[3] for row in rows) / 5
    totals = torch.zeros(5, 2)
    folds = [[] for _ in range(5)]
    capacities = [len(subjects) // 5 + (i < len(subjects) % 5) for i in range(5)]
    for _, _, subject, counts in sorted(rows, key=lambda row: row[:3], reverse=True):
        imbalance = (((totals + counts - target) / target.clamp_min(1)) ** 2).sum(dim=1)
        cost = [
            float(imbalance[i]) + 0.05 * (len(fold) + 1) ** 2
            if len(fold) < capacities[i] else float("inf")
            for i, fold in enumerate(folds)
        ]
        i = cost.index(min(cost))
        totals[i] += counts
        folds[i].append(subject)
    return folds


def fit_scaler(values, baseline=False):
    fill = torch.where(torch.isfinite(values), values, torch.nan).nanmean(dim=0).nan_to_num()
    clean = torch.where(torch.isfinite(values), values, fill)
    mean = clean.mean(dim=0)
    std = clean.std(dim=0, unbiased=False)
    if baseline:
        std = std.clamp_min(1e-6)
    else:
        std = torch.where(std <= 1e-12, 1.0, std)
    return fill, mean, std


def scale(values, scaler):
    fill, mean, std = scaler
    return (torch.where(torch.isfinite(values), values, fill) - mean) / std


def standardizers(graph, train_nodes, baseline):
    train_features = graph.x[train_nodes]
    if baseline:
        train_features = train_features.cpu()
    return (
        tuple(value.to(graph.x.device) for value in fit_scaler(train_features, baseline)),
        fit_scaler(graph.edge_attr[train_nodes[graph.edge_index[0]]]),
    )


def model_inputs(graph, scalers):
    node_scaler, edge_scaler = scalers
    return (scale(graph.x, node_scaler), graph.edge_index,
            scale(graph.edge_attr, edge_scaler), graph.graph_id)


def training_setup(name, seed, labels):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = MODELS[name]().to(labels.device)
    counts = torch.bincount(labels, minlength=2).float()
    if name in BASELINES:
        weights = torch.stack([counts.new_tensor(1), counts[0] / counts[1].clamp_min(1)])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    else:
        weights = counts.sum() / (2 * counts.clamp_min(1))
        weights /= weights.mean()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=3e-4)
    return model, weights, optimizer


def objective(logits, labels, weights, baseline):
    if baseline:
        return F.cross_entropy(logits, labels, weight=weights)
    true_logit = logits.gather(1, labels[:, None]).squeeze(1)
    other_logit = logits.gather(1, (1 - labels)[:, None]).squeeze(1)
    prob_true = logits.softmax(1).gather(1, labels[:, None]).squeeze(1)
    ce = F.cross_entropy(logits, labels, weight=weights, reduction="none")
    focal = ((1 - prob_true).clamp_min(1e-6) * ce).mean()
    margin = F.relu(0.20 - (true_logit - other_logit)).mean()
    return focal + 0.10 * margin


def train_epoch(model, inputs, mask, labels, weights, optimizer, baseline):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = objective(model(*inputs)[mask], labels, weights, baseline)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0 if baseline else 2.0)
    optimizer.step()
    return loss.item()


def metrics(labels, prob):
    labels, prob = labels.cpu(), prob.cpu()
    pred = prob.argmax(dim=1)
    cm = torch.bincount(2 * labels + pred, minlength=4).reshape(2, 2)
    precision = cm.diag() / cm.sum(dim=0).clamp_min(1)
    recall = cm.diag() / cm.sum(dim=1).clamp_min(1)
    f1 = 2 * cm.diag() / (cm.sum(dim=0) + cm.sum(dim=1)).clamp_min(1)
    result = {
        "accuracy": (pred == labels).float().mean().item(),
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
    }
    aps = []
    for i, label in enumerate(("negative", "target")):
        sorted_positive = labels[prob[:, i].argsort(descending=True)] == i
        ranks = torch.arange(1, len(labels) + 1)
        precision_at_k = sorted_positive.cumsum(0) / ranks
        ap = (precision_at_k * sorted_positive).sum() / sorted_positive.sum().clamp_min(1)
        ap = ap.item()
        result.update({f"{label}_precision": precision[i].item(),
                       f"{label}_recall": recall[i].item(), f"{label}_f1": f1[i].item(),
                       f"{label}_ap": ap})
        if sorted_positive.any():
            aps.append(ap)
    result["macro_ap"] = sum(aps) / len(aps)
    return result


def train_fold(name, graph, fold, subjects, epochs):
    baseline = name in BASELINES
    val_nodes = torch.isin(graph.subject_id, graph.subject_id.new_tensor(subjects))
    train = ~val_nodes & (graph.y >= 0)
    val = val_nodes & (graph.y >= 0)
    scalers = standardizers(graph, ~val_nodes, baseline)
    inputs = model_inputs(graph, scalers)
    seed = 43 + 100 * fold + BASELINES.index(name) if baseline else 41 + fold - 1
    model, weights, optimizer = training_setup(name, seed, graph.y[train])
    if not baseline:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-5,
        )

    history = []
    best_score = -float("inf")
    patience = 51 if baseline else 30
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, inputs, train, graph.y[train], weights, optimizer, baseline)
        model.eval()
        with torch.no_grad():
            logits = model(*inputs)
            prob = logits.softmax(1)
            val_loss = objective(logits[val], graph.y[val], weights, baseline)
            stats = metrics(graph.y[val], prob[val])
            train_stats = metrics(graph.y[train], prob[train])
        history.append({
            "fold": fold, "epoch": epoch, "loss": loss, "val_loss": val_loss.item(),
            "train_macro_f1": train_stats["macro_f1"], "val_macro_f1": stats["macro_f1"],
            "val_macro_precision": stats["macro_precision"],
        })
        if baseline:
            score = stats["target_ap"]
        else:
            score = min(stats["macro_precision"], stats["macro_recall"]) / 0.95
            score += 0.01 * stats["macro_f1"]
            scheduler.step(score)
        if score > best_score + 1e-5:
            best_score, best_epoch = score, epoch
            best_prob, best_stats = prob[val].cpu(), stats
        if epoch - best_epoch >= patience:
            break

    subject_ids = graph.subject_id[val].tolist()
    oof = pd.DataFrame({
        "fold": fold, "subject_id": [graph.subject_names[i] for i in subject_ids],
        "epoch_index": graph.epoch_index[val].tolist(),
        "channel_id": graph.channel_id[val].tolist(), "y_true": graph.y[val].tolist(),
        "p_negative": best_prob[:, 0], "p_target": best_prob[:, 1],
        "prediction": best_prob.argmax(dim=1),
    })
    result = {"fold": fold, "best_epoch": best_epoch, **best_stats}
    return result, history, oof


def cross_validate(name, graph, folds, epochs, output):
    results, history, predictions = [], [], []
    for number, subjects in enumerate(folds, start=1):
        result, epochs_run, oof = train_fold(name, graph, number, subjects, epochs)
        results.append(result)
        history.extend(epochs_run)
        predictions.append(oof)
        print(f"{name} fold {number}  macro F1 {result['macro_f1']:.4f}")
    fold_results = pd.DataFrame(results)
    oof = pd.concat(predictions, ignore_index=True)
    pooled = metrics(torch.tensor(oof.y_true.values),
                     torch.tensor(oof[["p_negative", "p_target"]].values))
    per_fold = fold_results[list(pooled)]
    summary = pd.DataFrame([per_fold.mean().to_dict(), per_fold.std().to_dict(), pooled],
                           index=["fold_mean", "fold_sd", "pooled_oof"])
    print(summary[["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_ap"]])
    if output is not None:
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        fold_results.to_csv(folder / "folds.csv", index=False)
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
        oof.to_csv(folder / "oof.csv", index=False)
        summary.to_csv(folder / "summary.csv", index_label="statistic")


def run_experiments(names, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=ROOT / "data/gnn_data.pt")
    parser.add_argument("--models", nargs="+", choices=names, default=names)
    parser.add_argument("--epochs", type=int, help="Maximum training epochs per fold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="Directory for metrics, history and OOF scores")
    args = parser.parse_args()
    raw = load_graph(args.data)
    graph = graph_without_bad_channels(raw)
    folds = subject_folds(graph)
    graph = graph.to(args.device)
    for name in args.models:
        epochs = args.epochs if args.epochs is not None else (200 if name in BASELINES else 80)
        if epochs < 1:
            parser.error("--epochs must be positive")
        cross_validate(name, graph, folds, epochs, args.output)
