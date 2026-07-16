from pathlib import Path

import numpy as np

from screening.addivortes_wrapper import (
    load_parameter_bounds_from_tex,
    make_parameter_simulator,
)


def test_make_parameter_simulator_maps_unit_cube_to_parameter_values():
    def score(params):
        return params["n_tessellations"] + params["burn_in"]

    simulate = make_parameter_simulator(
        score,
        parameter_names=["n_tessellations", "burn_in"],
        bounds={
            "n_tessellations": (10, 30),
            "burn_in": (50, 100),
        },
        baseline={"n_tessellations": 10, "burn_in": 50},
    )

    assert simulate(np.array([0.5, 0.5])) == 95.0
    assert simulate(np.array([1.0, 0.0])) == 80.0


def test_load_parameter_bounds_from_tex_reads_expected_ranges():
    tex_path = Path(__file__).resolve().parents[1] / ".." / "parameters.tex"
    bounds = load_parameter_bounds_from_tex(tex_path)

    assert bounds["m"] == (20.0, 500.0)
    assert bounds["nu"] == (1.0, 10.0)
    assert bounds["q"] == (0.6, 0.999)
    assert bounds["omega"] == (1.0, 7.0)
    assert bounds["lambda_c"] == (1.0, 50.0)
    assert bounds["iter"] == (500.0, 100000.0)
    assert bounds["burnin"] == (0.01, 0.5)
