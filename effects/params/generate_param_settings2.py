"""Generate the second parameter-effects settings CSV."""

from pathlib import Path

import generate_param_settings as generator


OUTPUT_PATH = Path(__file__).with_name("param_settings2.csv")
PARAMETER_VALUES = {
    "m": range(500, 5001, 100),
    "mcmcIter": range(2000, 100001, 2000),
}


def parameter_values(parameter_name: str):
    """Return the requested values for the parameter being varied."""
    return PARAMETER_VALUES[parameter_name]


def generate_param_settings2(output_path: Path = OUTPUT_PATH) -> Path:
    """Write the second parameter settings CSV and return its path."""
    return generator.generate_param_settings(
        output_path,
        parameter_names=list(PARAMETER_VALUES),
        value_provider=parameter_values,
    )


if __name__ == "__main__":
    path = generate_param_settings2()
    n_settings = sum(len(values) for values in PARAMETER_VALUES.values())
    print(f"Wrote {n_settings} settings to {path}")