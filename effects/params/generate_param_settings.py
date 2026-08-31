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
N_VALUES = 101

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
    "mcmcBurnin": 200,
}

SEARCH_RANGES = {
    "m": (20, 500),
    "nu": (1, 10),
    "q": (0.6, 0.999),
    "omega": (1, 7),
    "lambda": (1, 50),
    "mcmcIter": (500, 100000),
    "mcmcBurnin": (0.01, 0.5),
}

INTEGER_PARAMETERS = {"m", "nu", "omega", "lambda", "mcmcIter"}


def parameter_values(parameter_name: str):
    """Return the unique values used when varying one parameter."""
    lower, upper = SEARCH_RANGES[parameter_name]
    values = np.linspace(lower, upper, N_VALUES)
    if parameter_name in INTEGER_PARAMETERS:
        return np.unique(np.rint(values).astype(int))
    return np.round(values, 5)


def generate_param_settings(output_path: Path = OUTPUT_PATH) -> Path:
    """Write the one-at-a-time parameter settings CSV and return its path."""
    fieldnames = ["changed_parameter", *PARAMETER_NAMES]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for changed_parameter in PARAMETER_NAMES:
            for value in parameter_values(changed_parameter):
                settings = DEFAULTS.copy()
                settings[changed_parameter] = (
                    round(float(value), 5)
                    if changed_parameter not in INTEGER_PARAMETERS
                    else int(value)
                )
                if changed_parameter == "mcmcBurnin":
                    settings[changed_parameter] = int(
                        round(value * settings["mcmcIter"])
                    )
                writer.writerow(
                    {
                        "changed_parameter": changed_parameter,
                        **settings,
                    }
                )

    return output_path


if __name__ == "__main__":
    path = generate_param_settings()
    n_settings = sum(len(parameter_values(name)) for name in PARAMETER_NAMES)
    print(f"Wrote {n_settings} settings to {path}")