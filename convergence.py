# This function uses code from the articles below to give a benchmark test for run times on
# 4 different examples using the AddiVortes algorithm. Each example fits the model, and then
# finds the mean predictions for each data point.

from time import time

from addivortes import AddiVortesRegressor


def convergence(n_tesselations = 200, total_mcmc_iter = 2000, burn_in = 200):
    from addivortes import AddiVortesRegressor
    import pandas as pd
    import numpy as np
    
    def test1():
        x_train = pd.read_csv("./benchmarks/datasets/boston/x_train.csv")
        y_train = pd.read_csv("./benchmarks/datasets/boston/y_train.csv").iloc[:, 0].to_numpy()
        model = AddiVortesRegressor(
            n_tessellations = n_tesselations,
            total_mcmc_iter=total_mcmc_iter,
            burn_in = burn_in
        )
        model.fit(x_train, y_train)
        model.trace_diagnostics(
            plot_types=("trace", "histogram", "autocorrelation"),
            show=True
        )

    def test2():
        x_train = pd.read_csv("./benchmarks/datasets/synthetic/x_train.csv")
        y_train = pd.read_csv("./benchmarks/datasets/synthetic/y_train.csv").iloc[:, 0].to_numpy()
        model = AddiVortesRegressor(
            n_tessellations = n_tesselations,
            total_mcmc_iter=total_mcmc_iter,
            burn_in = burn_in
        )
        model.fit(x_train, y_train)
        model.trace_diagnostics(
            plot_types=("trace", "histogram", "autocorrelation"),
            show=True
        )

    def test3():
        x_train = pd.read_csv("./benchmarks/datasets/spherical/x_train.csv")
        y_train = pd.read_csv("./benchmarks/datasets/spherical/y_train.csv").iloc[:, 0].to_numpy()
        model = AddiVortesRegressor(
            n_tessellations = n_tesselations,
            total_mcmc_iter=total_mcmc_iter,
            burn_in = burn_in,
            metric = "spherical"
        )
        model.fit(x_train, y_train)
        model.trace_diagnostics(
            plot_types=("trace", "histogram", "autocorrelation"),
            show=True
        )
        
    def test4():
        x_train = pd.read_csv("./benchmarks/datasets/categorical/x_train.csv")
        y_train = pd.read_csv("./benchmarks/datasets/categorical/y_train.csv").iloc[:, 0].to_numpy()
        model = AddiVortesRegressor(
            n_tessellations = n_tesselations,
            total_mcmc_iter=total_mcmc_iter,
            burn_in = burn_in,
            cat_scaling = 1.0
        )
        model.fit(x_train, y_train)
        model.trace_diagnostics(
            plot_types=("trace", "histogram", "autocorrelation"),
            show=True
        )
    test1()
    test2()
    test3()
    test4()

convergence()