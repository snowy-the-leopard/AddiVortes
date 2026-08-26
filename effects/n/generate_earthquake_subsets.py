# Usage: python generate_earthquake_subsets.py
#
# Calls random_subset.py for each size 100, 200, ..., 10000. Each size n is
# repeated (10000 // n) times with a different random seed, so that the
# total rows sampled across repeats is always ~10000 (e.g. n=100 -> 100
# repeats, n=200 -> 50 repeats, ..., n=10000 -> 1 repeat).
#
# Output structure: subsets/<size>/<repeat>/train.csv and test.csv

import subprocess
import sys

INPUT_CSV = "benchmarks/datasets/earthquakes/earthquakes.csv"
OUTDIR_ROOT = "effects/n/subsets"
SIZES = range(100, 10001, 100)
TARGET_TOTAL_ROWS = 10000
BASE_SEED = 42

for n in SIZES:
    repeats = TARGET_TOTAL_ROWS // n
    for rep in range(repeats):
        outdir = f"{OUTDIR_ROOT}/{n}/{rep}"
        seed = BASE_SEED + rep
        subprocess.run([
            sys.executable, "effects/n/random_subset.py",
            INPUT_CSV,
            "--n", str(n),
            "--seed", str(seed),
            "--outdir", outdir,
        ], check=True)
