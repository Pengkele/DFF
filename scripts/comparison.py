"""Compare DFF with the 5 GADBench baselines."""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.training import BASELINES, run_experiments

    run_experiments(("DFF", *BASELINES), __doc__)
