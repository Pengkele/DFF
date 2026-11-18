"""Plot the training history and pooled out-of-fold confusion matrix in Figure 6."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_curve(ax, history, metric, label, color):
    summary = history.groupby("epoch")[metric].agg(["mean", "std", "count"])
    epoch = summary.index.to_numpy()
    mean = summary["mean"].to_numpy()
    ci = 1.96 * summary["std"].fillna(0).to_numpy() / summary["count"].to_numpy() ** 0.5
    ax.plot(epoch, mean, color=color, linewidth=2, label=label)
    ax.fill_between(
        epoch,
        mean - ci,
        mean + ci,
        color=color,
        alpha=0.2,
        linewidth=0,
        label=f"{label} 95% CI",
    )
    return (mean - ci).min()


def plot_training(history, oof):
    folds = history["fold"].nunique()
    fig, (loss, score, confusion) = plt.subplots(
        1, 3, figsize=(18, 4.7), width_ratios=[1.15, 1.15, 0.9], layout="constrained"
    )
    train_lower = plot_curve(loss, history, "loss", "Training loss", "#756bb1")
    val_lower = plot_curve(loss, history, "val_loss", "Validation loss", "#67a9cf")
    loss.set(
        title=f"{folds}-fold training and validation loss history",
        ylabel="Loss",
        ylim=(max(0, min(train_lower, val_lower) * 0.98), None),
    )

    lower = [
        plot_curve(score, history, "train_macro_f1", "Training macro F1", "#756bb1"),
        plot_curve(
            score,
            history,
            "val_macro_precision",
            "Validation macro precision",
            "#8073ac",
        ),
        plot_curve(score, history, "val_macro_f1", "Validation macro F1", "#67a9cf"),
    ]
    score.axhline(
        0.95, color="gray", linestyle=":", linewidth=1, label="Precision target"
    )
    score.set(
        title="Validation precision and train/validation F1 history",
        ylabel="Score",
        ylim=(max(0, min(lower) - 0.01), 1.01),
    )
    for ax in (loss, score):
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    counts = (
        pd.crosstab(oof["y_true"], oof["prediction"])
        .reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        .to_numpy()
    )
    image = confusion.imshow(counts, cmap="Blues")
    confusion.set(
        title=f"{folds}-fold OOF confusion matrix",
        xlabel="Predicted class",
        ylabel="True class",
        xticks=[0, 1],
        yticks=[0, 1],
    )
    for row in range(2):
        for col in range(2):
            confusion.text(
                col,
                row,
                str(counts[row, col]),
                ha="center",
                va="center",
                color="white" if image.norm(counts[row, col]) > 0.55 else "black",
            )
    fig.colorbar(image, ax=confusion, fraction=0.046, pad=0.04)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", type=Path, help="Model directory with history.csv and oof.csv"
    )
    parser.add_argument("--output", type=Path, help="Save the figure to this file")
    args = parser.parse_args()
    history = pd.read_csv(args.directory / "history.csv")
    oof = pd.read_csv(args.directory / "oof.csv")
    fig = plot_training(history, oof)
    if args.output:
        fig.savefig(args.output, dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
