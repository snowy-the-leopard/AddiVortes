import csv
import os
import time
import multiprocessing as mp
from pathlib import Path

# Belt-and-braces: also set these before numpy is imported, in case the
# platform's multiprocessing start method is "spawn" (Windows/macOS default),
# where each worker re-imports numpy fresh and these env vars are honored at
# BLAS init time. On "fork" (Linux default) numpy is already imported in the
# parent before workers exist, so this alone isn't sufficient there —
# threadpoolctl below (used in _init_worker) covers that case.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from addivortes import AddiVortesRegressor

FIELDNAMES = [
    "m", "nu", "q", "omega", "lambda_c", "iter", "burnin",
    "Fit time", "Prediction time", "In-sample RMSE", "Out-of-sample RMSE",
]

# Populated once per worker process by _init_worker(), not on every task.
_DATA = {}


def _init_worker(data_dir: str):
    """Runs once per worker process when the pool starts.

    Pins each worker's BLAS/OpenMP libraries to a single thread. Without
    this, if numpy is linked against a threaded BLAS (OpenBLAS/MKL), every
    one of the 128 worker *processes* could also spawn multiple internal
    BLAS *threads*, oversubscribing the cluster's cores many times over and
    tanking throughput. threadpoolctl sets this at runtime regardless of
    whether the worker process was created via fork or spawn, unlike the
    env vars above which only reliably help under spawn.
    """
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        # threadpoolctl isn't installed; the env vars set at module import
        # time still help under "spawn" start method, so this is a
        # best-effort fallback rather than a hard failure.
        pass

    data_dir = Path(data_dir)
    _DATA["x_train"] = pd.read_csv(data_dir / "x_train.csv")
    _DATA["y_train"] = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0].to_numpy()
    _DATA["x_test"] = pd.read_csv(data_dir / "x_test.csv")
    _DATA["y_test"] = pd.read_csv(data_dir / "y_test.csv").iloc[:, 0].to_numpy()


def grid_test(params: dict) -> dict:
    """Runs one grid configuration in a worker process. Returns a result row
    instead of writing to disk directly, so the parent process can serialize
    all writes and avoid concurrent-write corruption."""
    x_train = _DATA["x_train"]
    y_train = _DATA["y_train"]
    x_test = _DATA["x_test"]
    y_test = _DATA["y_test"]

    m, nu, q, omega, lambda_c, n_iter, burnin = (
        params["m"], params["nu"], params["q"], params["omega"],
        params["lambda_c"], params["iter"], params["burnin"],
    )

    start_time_fit = time.perf_counter()
    model = AddiVortesRegressor(
        n_tessellations=m,
        total_mcmc_iter=n_iter,
        burn_in=burnin,
        nu=nu,
        q=q,
        omega=omega,
        lambda_rate=lambda_c,
    )
    model.fit(x_train, y_train)
    end_time_fit = time.perf_counter()

    start_time_preds = time.perf_counter()
    preds = model.predict(x_test)
    end_time_preds = time.perf_counter()

    return {
        "m": m,
        "nu": nu,
        "q": q,
        "omega": omega,
        "lambda_c": lambda_c,
        "iter": n_iter,
        "burnin": burnin,
        "Fit time": round(end_time_fit - start_time_fit, 3),
        "Prediction time": round(end_time_preds - start_time_preds, 3),
        "In-sample RMSE": round(model.in_sample_rmse_, 3),
        "Out-of-sample RMSE": float(round(np.sqrt(np.mean((y_test - preds) ** 2)), 3)),
    }


def _grid_test_safe(params: dict):
    """Wraps grid_test so a failure in one setting (e.g. a numerical issue in
    the MCMC) is captured and returned rather than propagated as a raw
    exception, which is awkward to handle cleanly through
    Pool.imap_unordered. Returns (params, row_or_None, error_or_None)."""
    try:
        return params, grid_test(params), None
    except Exception as exc:
        return params, None, repr(exc)


PARAM_KEYS = ("m", "nu", "q", "omega", "lambda_c", "iter", "burnin")


def _param_key(params: dict) -> tuple:
    """Canonical key identifying a grid row, used to detect already-completed
    runs on resume. Floats are rounded to avoid string/precision mismatches
    between what was written to CSV and what's freshly parsed from grid.csv."""
    return (
        int(params["m"]),
        int(params["nu"]),
        round(float(params["q"]), 6),
        int(params["omega"]),
        int(params["lambda_c"]),
        int(params["iter"]),
        int(params["burnin"]),
    )


def load_completed_keys(output_path: Path) -> set:
    """Reads whatever rows already exist in the output log (from a previous,
    possibly interrupted run) so we can skip redoing them."""
    completed = set()
    if not output_path.exists() or output_path.stat().st_size == 0:
        return completed
    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(_param_key(row))
            except (KeyError, ValueError):
                # Malformed/partial trailing row (e.g. from a crash mid-write);
                # skip it rather than fail the whole run.
                continue
    return completed


def load_param_grid(grid_csv_path: Path) -> list[dict]:
    rows = []
    with grid_csv_path.open("r", newline="", encoding="utf-8") as settings_file:
        reader = csv.reader(settings_file)
        next(reader)  # skip header row
        for m, nu, q, omega, lambda_c, n_iter, burnin in reader:
            n_iter = int(n_iter)
            rows.append({
                "m": int(m),
                "nu": int(nu),
                "q": float(q),
                "omega": int(omega),
                "lambda_c": int(lambda_c),
                "iter": n_iter,
                "burnin": int(round(float(burnin) * n_iter)),
            })
    return rows


def default_n_workers() -> int:
    """Picks a worker count appropriate to how the job was launched.
    Respects SLURM's allocation if present, otherwise falls back to all
    visible cores, so the same script works both under sbatch/srun and when
    run manually."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    return os.cpu_count() or 1


def grid_tests(n_workers: int | None = None, resume: bool = True):
    here = Path(__file__).parent
    data_dir = here / "benchmarks" / "datasets" / "boston"
    output_path = here / "grid-testlog.csv"
    grid_csv_path = here / "grid.csv"

    if n_workers is None:
        n_workers = default_n_workers()

    full_grid = load_param_grid(grid_csv_path)

    completed_keys = load_completed_keys(output_path) if resume else set()
    param_grid = [p for p in full_grid if _param_key(p) not in completed_keys]

    n_skipped = len(full_grid) - len(param_grid)
    if n_skipped:
        print(f"Resuming: skipping {n_skipped} already-completed rows found in {output_path.name}")

    if not param_grid:
        print("Nothing to do: all grid rows already completed.")
        return

    # Only write the header if the file is new/empty; in resume mode we're
    # appending to an existing log, so must never truncate it.
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    if write_header:
        with output_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # One worker process per setting, capped at n_workers (e.g. 128 cores).
    # With exactly 128 settings and 128 cores, this puts every setting on
    # its own core simultaneously; with more settings than cores, the pool
    # queues the remainder and picks them up as workers free up.
    n_workers = min(n_workers, len(param_grid)) or 1
    print(f"Running {len(param_grid)} grid rows across {n_workers} worker processes...")

    # chunksize=1: each worker pulls one setting at a time from the queue.
    # Settings have widely varying MCMC cost (m, iter, burnin all vary), so
    # a larger chunksize could let one worker get stuck with several slow
    # settings while others sit idle; chunksize=1 keeps the pool balanced.
    with mp.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(str(data_dir),),
    ) as pool:
        results_iter = pool.imap_unordered(_grid_test_safe, param_grid, chunksize=1)

        # Always append here: header (if any) was already written above, and
        # in resume mode the file may already contain prior rows.
        with output_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            for i, (params, row, error) in enumerate(results_iter, 1):
                if error is not None:
                    print(f"[FAILED] params={params} -> {error}")
                    continue
                writer.writerow(row)
                csv_file.flush()  # persist incrementally in case of crash/preemption
                print(f"[{i}/{len(param_grid)}] done: {row}")


if __name__ == "__main__":
    grid_tests()