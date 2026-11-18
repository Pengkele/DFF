"""Run the feature, graph and architecture ablations in Table 3."""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model import MODELS
    from scripts.training import run_experiments

    run_experiments(tuple(name for name in MODELS if name.startswith("DFF")), __doc__)
