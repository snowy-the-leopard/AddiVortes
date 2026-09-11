"""Generate one-at-a-time AddiVortes parameter settings.

Each parameter is varied over up to 101 equally spaced values while every
other parameter remains at its documented default. Integer values are
deduplicated when their range contains fewer than 101 possible values. The
burn-in search range is
specified as a proportion, then converted to an integer count using the
number of MCMC iterations before it is written to the CSV.
"""

import csv
from pathlib import Path

import numpy as np


OUTPUT_PATH = Path(__file__).with_name("param_settings.csv")

PARAMETER_NAMES = [
    "m",
    "nu",
    "q",
    "omega",
    "lambda",
    "mcmcIter",
    "mcmcBurnin",
]

DEFAULTS = {
    "m": 200,
    "nu": 6,
    "q": 0.85,
    "omega": 3,
    "lambda": 25,
    "mcmcIter": 1200,
    "mcmcBurnin": 0.1666,
}

NEW_DEFAULTS = {
    "m": 500,
    "omega": 1,
    "lambda": 1,
    "mcmcIter": 5000
}


def generate_param_settings(output_path: Path = OUTPUT_PATH) -> Path:
    """Write the one-at-a-time parameter settings CSV and return its path."""
    fieldnames = ["test_no.", *PARAMETER_NAMES]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(0, 16):
            inclusion = "00000" + bin(i)[2:] 
            print(inclusion)
            lam = DEFAULTS["lambda"] if inclusion[-1] == "0" else NEW_DEFAULTS["lambda"]
            m = DEFAULTS["m"] if inclusion[-2] == "0" else NEW_DEFAULTS["m"]
            mcmcIter = DEFAULTS["mcmcIter"] if inclusion[-3] == "0" else NEW_DEFAULTS["mcmcIter"]
            omega = DEFAULTS["omega"] if inclusion[-4] == "0" else NEW_DEFAULTS["omega"]
            writer.writerow(
                {
                    "test_no.": i+1,
                    "m": m,
                    "nu": DEFAULTS["nu"],
                    "q": DEFAULTS["q"],
                    "omega": omega,
                    "lambda": lam,
                    "mcmcIter": mcmcIter,
                    "mcmcBurnin": DEFAULTS["mcmcBurnin"],
                }
            )
    return output_path


if __name__ == "__main__":
    path = generate_param_settings()
    print(f"Wrote 16 settings to {path}")