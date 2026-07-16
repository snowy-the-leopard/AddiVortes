"""Minimal example of screening a scalar objective over AddiVortes parameters."""

from pathlib import Path

import matplotlib.pyplot as plt

from plotting import plot_final_effects, plot_iteration_grid
from sequential_ee import sequential_ee_screen
from addivortes_wrapper import load_parameter_bounds_from_tex, make_parameter_simulator


def objective(params: dict[str, float]) -> float:
    return (
        abs(params["m"] - 200) / 100.0
        + abs(params["nu"] - 6.0) / 10.0
        + abs(params["q"] - 0.85)
        + abs(params["omega"] - 3.0) / 5.0
        + abs(params["lambda_c"] - 25.0) / 25.0
        + abs(params["iter"] - 1200.0) / 1000.0
        + abs(params["burnin"] - 0.1666666667)
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    bounds = load_parameter_bounds_from_tex(root / "parameters.tex")
    parameter_names = ["m", "nu", "q", "omega", "lambda_c", "iter", "burnin"]

    simulate = make_parameter_simulator(
        objective,
        parameter_names=parameter_names,
        bounds={name: bounds[name] for name in parameter_names},
    )

    result = sequential_ee_screen(
        simulate,
        k=len(parameter_names),
        m=8,
        delta=0.556,
        gamma=0.01,
        seed=1,
        factor_names=parameter_names,
    )

    print(result.summary())
    plot_final_effects(result)
    plot_iteration_grid(result)
    plt.show()


if __name__ == "__main__":
    main()
