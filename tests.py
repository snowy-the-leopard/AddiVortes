# This function uses code from the articles below to give a benchmark test for run times on
# 4 different examples using the AddiVortes algorithm. Each example fits the model, and then
# finds the mean predictions for each data point.

def tests(nDigits = 3):
    from addivortes import AddiVortesRegressor
    import pandas as pd
    import time
    import numpy as np

    nDigits = 3

    def test1():
        x_train = pd.read_csv("./datasets/boston/x_train.csv")
        y_train = pd.read_csv("./datasets/boston/y_train.csv").iloc[:, 0].to_numpy()
        x_test = pd.read_csv("./datasets/boston/x_test.csv")
        y_test = pd.read_csv("./datasets/boston/y_test.csv").iloc[:, 0].to_numpy()

        start_time_fit = time.perf_counter()
        model = AddiVortesRegressor(
            n_tessellations = 200,
            total_mcmc_iter=2000,
            burn_in = 200
        )
        model.fit(x_train, y_train)
        end_time_fit = time.perf_counter()

        start_time_preds = time.perf_counter()
        preds = model.predict(x_test)
        end_time_preds = time.perf_counter()

        d = {"Fit time": [round(end_time_fit-start_time_fit, nDigits)],
              "Prediction time": [round(end_time_preds-start_time_preds, nDigits)],
              "In-sample RMSE": [round(model.in_sample_rmse_, nDigits)],
              "Out-of-sample RMSE": [float(round(np.sqrt(np.mean((y_test - preds) ** 2)), nDigits))]}
        return(pd.DataFrame(data=d))

    def test2():
        x_train = pd.read_csv("./datasets/synthetic/x_train.csv")
        y_train = pd.read_csv("./datasets/synthetic/y_train.csv").iloc[:, 0].to_numpy()
        x_test = pd.read_csv("./datasets/synthetic/x_test.csv")
        y_test = pd.read_csv("./datasets/synthetic/y_test.csv").iloc[:, 0].to_numpy()

        start_time_fit = time.perf_counter()
        model = AddiVortesRegressor(n_tessellations = 50)
        model.fit(x_train, y_train)
        end_time_fit = time.perf_counter()

        start_time_preds = time.perf_counter()
        preds = model.predict(x_test)
        end_time_preds = time.perf_counter()

        d = {"Fit time": [round(end_time_fit-start_time_fit, nDigits)],
              "Prediction time": [round(end_time_preds-start_time_preds, nDigits)],
              "In-sample RMSE": [round(model.in_sample_rmse_, nDigits)],
              "Out-of-sample RMSE": [float(round(np.sqrt(np.mean((y_test - preds) ** 2)), nDigits))]}
        return(pd.DataFrame(data=d))

    def test3():
        x_train = pd.read_csv("./datasets/spherical/x_train.csv")
        y_train = pd.read_csv("./datasets/spherical/y_train.csv").iloc[:, 0].to_numpy()
        x_test = pd.read_csv("./datasets/spherical/x_test.csv")
        y_test = pd.read_csv("./datasets/spherical/y_test.csv").iloc[:, 0].to_numpy()

        start_time_fit = time.perf_counter()
        model = AddiVortesRegressor(
            n_tessellations = 50,
            total_mcmc_iter=500, # Using default R values for mcmc iter and burn in
            burn_in = 100,
            metric="spherical"
        )
        model.fit(x_train, y_train)
        end_time_fit = time.perf_counter()

        start_time_preds = time.perf_counter()
        preds = model.predict(x_test)
        end_time_preds = time.perf_counter()

        d = {"Fit time": [round(end_time_fit-start_time_fit, nDigits)],
              "Prediction time": [round(end_time_preds-start_time_preds, nDigits)],
              "In-sample RMSE": [round(model.in_sample_rmse_, nDigits)],
              "Out-of-sample RMSE": [float(round(np.sqrt(np.mean((y_test - preds) ** 2)), nDigits))]}
        return(pd.DataFrame(data=d))

    def test4():
        x_train = pd.read_csv("./datasets/categorical/x_train.csv")
        y_train = pd.read_csv("./datasets/categorical/y_train.csv").iloc[:, 0].to_numpy()
        x_test = pd.read_csv("./datasets/categorical/x_test.csv")
        y_test = pd.read_csv("./datasets/categorical/y_test.csv").iloc[:, 0].to_numpy()

        start_time_fit = time.perf_counter()
        model = AddiVortesRegressor(
            n_tessellations = 50,
            total_mcmc_iter=500, # Using default R values for mcmc iter and burn in
            burn_in = 100,
            cat_scaling=1.0
            )
        model.fit(x_train, y_train)
        end_time_fit = time.perf_counter()

        start_time_preds = time.perf_counter()
        preds = model.predict(x_test)
        end_time_preds = time.perf_counter()

        d = {"Fit time": [round(end_time_fit-start_time_fit, nDigits)],
              "Prediction time": [round(end_time_preds-start_time_preds, nDigits)],
              "In-sample RMSE": [round(model.in_sample_rmse_, nDigits)],
              "Out-of-sample RMSE": [float(round(np.sqrt(np.mean((y_test - preds) ** 2)), nDigits))]}
        return(pd.DataFrame(data=d))

    results = pd.concat([test1(), test2(), test3(), test4()], axis=0)
    results.index = ["Test 1", "Test 2", "Test 3", "Test 4"]
    times = results.iloc[:, :2]
    errors = results.iloc[:, 2:]
    times["Sum"] = times.sum(axis=1)
    times.loc["Sum"] = times.sum(axis=0)
    errors["Sum"] = errors.sum(axis=1)
    errors.loc["Sum"] = errors.sum(axis=0)
    return([times, errors])

results = tests()
print(results[0])
print(results[1])