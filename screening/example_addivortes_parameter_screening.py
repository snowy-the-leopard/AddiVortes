"""
Example: screen AddiVortes hyperparameters with sequential elementary effects.

This script treats the AddiVortes regressor itself as the black-box function
being screened. Each screened input is a hyperparameter, and the objective is
an out-of-sample RMSE score computed after fitting the model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from addivortes import AddiVortesRegressor

from plotting import plot_final_effects, plot_iteration_grid
from sequential_ee import sequential_ee_screen
from addivortes_wrapper import load_parameter_bounds_from_tex, make_parameter_simulator

import matplotlib.pyplot as plt


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "benchmarks" / "datasets" / "boston"

    X_train = pd.read_csv(data_dir / "x_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0].to_numpy()
    X_test = pd.read_csv(data_dir / "x_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").iloc[:, 0].to_numpy()

    parameter_names = ["m", "nu", "q", "omega", "lambda_c", "iter", "burnin"]
    bounds = load_parameter_bounds_from_tex(root / "parameters.tex")
    bounds = {
        name: bounds[name]
        for name in parameter_names
        if name in bounds
    }
    baseline = {
        "m": 200.0,
        "nu": 6.0,
        "q": 0.85,
        "omega": 3.0,
        "lambda_c": 25.0,
        "iter": 1200.0,
        "burnin": 0.1666666667,
    }

    def fit_and_score(params: dict[str, float]) -> float:
        total_iter = int(round(params["iter"]))
        burn_in = int(round(params["burnin"] * total_iter))
        model = AddiVortesRegressor(
            n_tessellations=int(round(params["m"])),
            total_mcmc_iter=total_iter,
            burn_in=burn_in,
            nu=float(params["nu"]),
            q=float(params["q"]),
            omega=float(params["omega"]),
            lambda_rate=float(params["lambda_c"]),
            random_state=0,
            verbose=False,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return float(np.sqrt(np.mean((y_test - preds) ** 2)))

    simulate = make_parameter_simulator(
        fit_and_score,
        parameter_names=parameter_names,
        bounds=bounds,
        baseline=baseline,
    )

    p = 10
    delta = p / (2 * (p - 1))
    gamma = 0.01

    result = sequential_ee_screen(
        simulate,
        k=len(parameter_names),
        m=6,
        delta=delta,
        gamma=gamma,
        seed=1,
        factor_names=parameter_names,
    )

    print(result.summary())
    print()
    print("Linear/negligible parameters:", [parameter_names[i] for i in result.linear_factors])
    print("Nonlinear/interacting parameters:", [parameter_names[i] for i in result.nonlinear_factors])

    plot_final_effects(result)
    plot_iteration_grid(result)
    plt.show()


if __name__ == "__main__":
    main()
