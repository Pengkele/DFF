"""Refit on the development cohort and predict channels in held-out subjects."""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.training import (
    BASELINES, ROOT, graph_without_bad_channels, load_graph, model_inputs,
    standardizers, train_epoch, training_setup,
)


MODELS_FOR_LOCALIZATION = ("DFF", "GraphSAGE", "DFF-noSpatialGradient")


def fit_and_predict(name, train, held_out, epochs):
    baseline = name in BASELINES
    all_nodes = torch.ones_like(train.y, dtype=torch.bool)
    scalers = standardizers(train, all_nodes, baseline)
    inputs = model_inputs(train, scalers)
    labelled = train.y >= 0
    labels = train.y[labelled]
    seed = {"DFF": 41, "GraphSAGE": 443, "DFF-noSpatialGradient": 841}[name]
    model, weights, optimizer = training_setup(name, seed, labels)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    for _ in range(epochs):
        train_epoch(model, inputs, labelled, labels, weights, optimizer, baseline)
        scheduler.step()

    model.eval()
    with torch.no_grad():
        logits = model(*model_inputs(held_out, scalers))
        return logits.softmax(dim=1).cpu()


def predictions_by_channel(raw, name, prob):
    keep = raw.channel_flag.reshape(-1) != -1
    target_score = torch.full((len(keep),), float("nan"))
    target_score[keep] = prob[:, 1]
    flags = torch.full_like(raw.channel_flag.reshape(-1), -1)
    flags[keep] = 2 * prob.argmax(dim=1).to(flags.dtype)
    return pd.DataFrame({
        "model": name, "subject_id": raw.subject_id.tolist(),
        "subject_name": [raw.subject_names[i] for i in raw.subject_id.tolist()],
        "epoch_index": raw.epoch_index.tolist(),
        "epoch_field": [raw.graph_epoch_fields[i] for i in raw.graph_id.tolist()],
        "channel_id": raw.channel_id.tolist(), "channel_flag": raw.channel_flag.reshape(-1).tolist(),
        "p_target": target_score, "predicted_flag": flags,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/gnn_data.pt")
    parser.add_argument("--held-out", type=Path, default=ROOT / "data/val_data/gnn_data.pt")
    parser.add_argument("--models", nargs="+", choices=MODELS_FOR_LOCALIZATION,
                        default=MODELS_FOR_LOCALIZATION)
    parser.add_argument("--epochs", type=int, help="Number of full-cohort training epochs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, help="Destination CSV for per-channel predictions")
    args = parser.parse_args()
    train_raw = load_graph(args.train)
    held_out_raw = load_graph(args.held_out)
    if set(train_raw.subject_names).intersection(held_out_raw.subject_names):
        parser.error("Development and held-out subject identifiers overlap.")
    train = graph_without_bad_channels(train_raw).to(args.device)
    held_out = graph_without_bad_channels(held_out_raw).to(args.device)
    predictions = []
    for name in args.models:
        epochs = args.epochs if args.epochs is not None else (200 if name in BASELINES else 80)
        if epochs < 1:
            parser.error("--epochs must be positive")
        prob = fit_and_predict(name, train, held_out, epochs)
        predictions.append(predictions_by_channel(held_out_raw, name, prob))
    frame = pd.concat(predictions, ignore_index=True)
    if args.output is None:
        frame.to_csv(sys.stdout, index=False)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
