def grid_tests():
    import csv
    from pathlib import Path

    from addivortes import AddiVortesRegressor
    import numpy as np
    import pandas as pd
    import time

    x_train = pd.read_csv("./benchmarks/datasets/boston/x_train.csv")
    y_train = pd.read_csv("./benchmarks/datasets/boston/y_train.csv").iloc[:, 0].to_numpy()
    x_test = pd.read_csv("./benchmarks/datasets/boston/x_test.csv")
    y_test = pd.read_csv("./benchmarks/datasets/boston/y_test.csv").iloc[:, 0].to_numpy()

    output_path = Path(__file__).with_name("grid-testlog.csv")
    fieldnames = ["m", "nu", "q", "omega", "lambda_c", "iter", "burnin", "Fit time", "Prediction time", "In-sample RMSE", "Out-of-sample RMSE"]

    def grid_test(m, nu, q, omega, lambda_c, iter, burnin):
        start_time_fit = time.perf_counter()
        model = AddiVortesRegressor(
            n_tessellations=m,
            total_mcmc_iter=iter,
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

        row = {
            "m": m,
            "nu": nu,
            "q": q,
            "omega": omega,
            "lambda_c": lambda_c,
            "iter": iter,
            "burnin": burnin,
            "Fit time": round(end_time_fit - start_time_fit, 3),
            "Prediction time": round(end_time_preds - start_time_preds, 3),
            "In-sample RMSE": round(model.in_sample_rmse_, 3),
            "Out-of-sample RMSE": float(round(np.sqrt(np.mean((y_test - preds) ** 2)), 3)),
        }

        with output_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if output_path.stat().st_size == 0:
                writer.writeheader()
            writer.writerow(row)

    with Path(__file__).with_name("grid.csv").open("r", newline="", encoding="utf-8") as settings_file:
        reader = csv.reader(settings_file)
        next(reader)  # skip header row
        for m, nu, q, omega, lamba_c, iter, burnin in reader:
            grid_test(int(m), int(nu), float(q), int(omega), int(lamba_c), int(iter), int(round(float(burnin)*int(iter))))

grid_tests()