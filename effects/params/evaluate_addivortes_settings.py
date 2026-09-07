"""Evaluate one AddiVortes model for each row in param_settings.csv."""

import csv
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from addivortes import AddiVortesRegressor

SETTINGS_PATH = Path(__file__).with_name("param_settings.csv")
RESULTS_PATH = Path(__file__).with_name("settings_results_full.csv")
PARENT = Path(__file__).resolve().parent.parent.parent
X_TRAIN = PARENT / "benchmarks" / "datasets" / "earthquakes" / "x_train.csv"
X_TEST = PARENT / "benchmarks" / "datasets" / "earthquakes" / "x_test.csv"
Y_TRAIN = PARENT / "benchmarks" / "datasets" / "earthquakes" / "y_train.csv"
Y_TEST = PARENT / "benchmarks" / "datasets" / "earthquakes" / "y_test.csv"
TARGET_COL = "mag"

PARAMETER_NAMES = [
    "m", "nu", "q", "omega", "lambda", "mcmcIter", "mcmcBurnin",
]
RESULT_NAMES = [
    "Fit time", "Prediction time", "In-sample RMSE", "Out-of-sample RMSE",
]
_DATA = {}


def _init_worker(train_path: str, test_path: str):
    """Load shared data and limit numerical libraries to one thread."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        pass

    _DATA["x_train"] = pd.read_csv(train_path)
    _DATA["y_train"] = pd.read_csv(Y_TRAIN).iloc[:, 0].to_numpy()
    _DATA["x_test"] = pd.read_csv(test_path)
    _DATA["y_test"] = pd.read_csv(Y_TEST).iloc[:, 0].to_numpy()


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def evaluate_setting(index_and_settings: tuple[int, dict]) -> tuple[int, dict]:
    """Fit and evaluate one parameter-settings row in a worker process."""
    index, settings = index_and_settings
    model = AddiVortesRegressor(
        n_tessellations=int(settings["m"]),
        total_mcmc_iter=int(settings["mcmcIter"]),
        burn_in=int(settings["mcmcBurnin"]),
        nu=int(settings["nu"]),
        q=float(settings["q"]),
        omega=int(settings["omega"]),
        lambda_rate=int(settings["lambda"]),
    )

    fit_start = time.perf_counter()
    model.fit(_DATA["x_train"], _DATA["y_train"])
    fit_time = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    train_predictions = model.predict(_DATA["x_train"])
    test_predictions = model.predict(_DATA["x_test"])
    predict_time = time.perf_counter() - predict_start

    result = {
        **settings,
        "Fit time": round(fit_time, 3),
        "Prediction time": round(predict_time, 3),
        "In-sample RMSE": round(_rmse(_DATA["y_train"], train_predictions), 3),
        "Out-of-sample RMSE": round(_rmse(_DATA["y_test"], test_predictions), 3),
    }
    return index, result


def _evaluate_setting_safe(index_and_settings: tuple[int, dict]):
    """Return a row-level error instead of stopping the whole pool."""
    index = index_and_settings[0]
    try:
        return index, evaluate_setting(index_and_settings)[1], None
    except Exception as exc:
        return index, None, repr(exc)


def default_n_workers() -> int:
    """Use the scheduler allocation when available, otherwise visible cores."""
    for variable in ("SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_CPUS_ON_NODE"):
        value = os.environ.get(variable)
        if value:
            try:
                return int(value)
            except ValueError:
                pass

    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def load_completed_settings(results_path: Path, parameter_names: list[str]) -> set:
    """Return parameter rows already written to the results CSV."""
    if not results_path.exists() or results_path.stat().st_size == 0:
        return set()

    with results_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return {
            tuple(row[name] for name in parameter_names)
            for row in reader
            if all(name in row for name in parameter_names)
        }


def evaluate_settings(
    settings_path: Path = SETTINGS_PATH,
    results_path: Path = RESULTS_PATH,
    train_path: Path = X_TRAIN,
    test_path: Path = X_TEST,
    n_workers: int | None = None,
):
    """Evaluate settings and append each completed result to a separate CSV."""
    settings_path = Path(settings_path)
    results_path = Path(results_path)
    with settings_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    fieldnames = list(rows[0]) + RESULT_NAMES
    parameter_names = list(rows[0])
    completed_settings = load_completed_settings(results_path, parameter_names)
    jobs = [
        (index, row)
        for index, row in enumerate(rows)
        if tuple(row[name] for name in parameter_names) not in completed_settings
    ]
    skipped = len(rows) - len(jobs)
    if skipped:
        print(f"Resuming: skipping {skipped} completed settings.")
    if not jobs:
        print("Nothing to do: all settings are already complete.")
        return

    if n_workers is None:
        n_workers = default_n_workers()
    n_workers = min(n_workers, len(jobs)) or 1

    write_header = not results_path.exists() or results_path.stat().st_size == 0
    with results_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        with mp.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(str(train_path), str(test_path)),
        ) as pool:
            result_iter = pool.imap_unordered(
                _evaluate_setting_safe,
                jobs,
                chunksize=1,
            )
            for completed, (index, row, error) in enumerate(result_iter, 1):
                if error is not None:
                    print(f"[FAILED] row={index + 2} -> {error}")
                    continue
                writer.writerow(row)
                csv_file.flush()
                os.fsync(csv_file.fileno())
                print(f"[{completed}/{len(jobs)}] done: row={index + 2}")


if __name__ == "__main__":
    evaluate_settings()
