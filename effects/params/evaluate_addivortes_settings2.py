"""Evaluate the second parameter-effects settings sweep.

Each completed row is appended and synced by the shared evaluator immediately,
so completed work remains available for resuming after an interruption.
"""

from pathlib import Path

import evaluate_addivortes_settings as evaluator


SETTINGS_PATH = Path(__file__).with_name("param_settings2.csv")
RESULTS_PATH = Path(__file__).with_name("settings_results2.csv")
N_WORKERS = 96


def evaluate_settings(
    settings_path: Path = SETTINGS_PATH,
    results_path: Path = RESULTS_PATH,
    n_workers: int = N_WORKERS,
):
    """Run the sweep, flushing each completed result for resumability."""
    return evaluator.evaluate_settings(
        settings_path=settings_path,
        results_path=results_path,
        n_workers=n_workers,
    )


if __name__ == "__main__":
    evaluate_settings()