"""Write predicted channel selections into per-trial Brainstorm MAT files."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat


def export_trials(subjects, predictions, output, legacy_flags=False):
    for subject, rows in predictions.groupby("subject_name"):
        directory = subjects / subject
        channels = loadmat(directory / "chan.mat", squeeze_me=True, struct_as_record=False)["chan"].Channel
        meg = np.array([i for i, channel in enumerate(np.atleast_1d(channels)) if channel.Type == "MEG"])
        epochs = loadmat(directory / "data.mat")
        for (model, field), scores in rows.groupby(["model", "epoch_field"]):
            source = epochs[field][0, 0]
            trial = {name: source[name] for name in source.dtype.names}
            flags = trial["ChannelFlag"].copy()
            scores = scores.set_index("channel_id").loc[np.arange(len(meg)), "predicted_flag"]
            flags.flat[meg] = scores.replace({2: 3}) if legacy_flags else scores
            trial["ChannelFlag"] = flags
            comment = str(np.asarray(trial.get("Comment", "event1_dff")).ravel()[0])
            event = re.sub(r"\s*\(#\d+\)\s*$", "", comment)
            event = re.sub(r"[^\w-]+", "_", event).strip("_") or "event1_dff"
            index = int(field.split("_")[-1])
            destination = output / subject / model
            destination.mkdir(parents=True, exist_ok=True)
            savemat(destination / f"data_{event}_trial{index:03d}.mat", trial)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subjects", type=Path, help="Validation folder containing sub-*/data.mat")
    parser.add_argument("predictions", type=Path, help="CSV written by predict.py")
    parser.add_argument("output", type=Path)
    parser.add_argument("--legacy-flags", action="store_true", help="Write selected channels as 3 for original DFF exports")
    args = parser.parse_args()
    export_trials(args.subjects, pd.read_csv(args.predictions), args.output, args.legacy_flags)
