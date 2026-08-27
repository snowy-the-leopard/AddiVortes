# Usage: python generate_earthquake_subsets_multiprocessing.py
#
# Calls random_subset.py for each size 100, 200, ..., 10000. Each size n is
# repeated (100000 // n) times with a different random seed, so that the
# total rows sampled across repeats is always ~100000 (e.g. n=100 -> 1000
# repeats, n=200 -> 500 repeats, ..., n=10000 -> 10 repeats).
#
# One batch = all repeats for a single value of n, run together on one
# worker process. This matches the multiprocessing strategy used by
# evaluate_addivortes_subsets.py.
#
# Output structure: subsets/<size>/<repeat>/train.csv and test.csv

import multiprocessing as mp
import os
import subprocess
import sys

INPUT_CSV = "benchmarks/datasets/earthquakes/earthquakes.csv"
OUTDIR_ROOT = "effects/n/subsets"
SIZES = range(100, 10001, 100)
TARGET_TOTAL_ROWS = 100000
BASE_SEED = 42


def _init_worker():
    """Runs once per worker process when the pool starts."""
    pass


def process_batch(n: int) -> dict:
    """Generate all repeats for one value of n in a worker process."""
    repeats = TARGET_TOTAL_ROWS // n

    for rep in range(repeats):
        outdir = f"{OUTDIR_ROOT}/{n}/{rep}"
        seed = BASE_SEED + rep

        subprocess.run(
            [
                sys.executable,
                "effects/n/random_subset.py",
                INPUT_CSV,
                "--n",
                str(n),
                "--seed",
                str(seed),
                "--outdir",
                outdir,
            ],
            check=True,
        )

    return {
        "n": n,
        "repeats": repeats,
    }


def _process_batch_safe(n: int):
    """Capture failures from an individual n rather than stopping the pool."""
    try:
        return n, process_batch(n), None
    except Exception as exc:
        return n, None, repr(exc)


def default_n_workers() -> int:
    """Use the SLURM allocation when available, otherwise visible CPU cores."""
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val:
            try:
                return int(val)
            except ValueError:
                pass

    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def generate_subsets(n_workers: int | None = None):
    sizes = list(SIZES)

    if n_workers is None:
        n_workers = default_n_workers()

    n_workers = min(n_workers, len(sizes)) or 1

    print(
        f"Generating {len(sizes)} batches across "
        f"{n_workers} worker processes..."
    )

    # Each worker takes one n at a time. This keeps the pool balanced because
    # smaller n values have many more repeats than larger n values.
    with mp.Pool(processes=n_workers, initializer=_init_worker) as pool:
        results_iter = pool.imap_unordered(
            _process_batch_safe,
            sizes,
            chunksize=1,
        )

        for i, (n, result, error) in enumerate(results_iter, 1):
            if error is not None:
                print(f"[FAILED] n={n} -> {error}")
                continue

            print(
                f"[{i}/{len(sizes)}] done: "
                f"n={result['n']}, repeats={result['repeats']}"
            )


if __name__ == "__main__":
    generate_subsets()
