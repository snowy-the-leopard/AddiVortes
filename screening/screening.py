"""
Example: screen AddiVortes hyperparameters with sequential elementary effects.

This script screens AddiVortes regressor over multiple metrics:
- In-sample RMSE
- Out-of-sample RMSE
- Fit time
- Predict time

Each screened input is a hyperparameter, and each metric is screened separately
to identify which parameters significantly affect that metric.

Parallelization:
- 4 metrics run in parallel across separate worker processes
- Plotting for all metrics runs in parallel after screening
"""

from __future__ import annotations

import csv
import os
import time
import multiprocessing as mp
from pathlib import Path

# Pin BLAS/OpenMP to single thread per worker process to avoid oversubscription
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from addivortes import AddiVortesRegressor

from plotting import plot_final_effects, plot_iteration_grid
from sequential_ee import sequential_ee_screen
from addivortes_wrapper import load_parameter_bounds_from_tex, make_parameter_simulator

import matplotlib.pyplot as plt


def _init_worker():
    """Runs once per worker process when the pool starts.
    
    Pins each worker's BLAS/OpenMP libraries to a single thread to prevent
    oversubscription when multiple worker processes each try to use threaded BLAS.
    """
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except ImportError:
        # threadpoolctl isn't installed; the env vars set at module import
        # time still help under "spawn" start method, so this is a best-effort
        # fallback rather than a hard failure.
        pass


def screen_one_metric(args: tuple) -> dict:
    """Screen one metric in a worker process.
    
    Args:
        args: (metric, X_train, y_train, X_test, y_test, parameter_names, 
               bounds, baseline, p, gamma)
    
    Returns:
        Dictionary with metric, result object, and timing information.
    """
    (metric, X_train, y_train, X_test, y_test, parameter_names, 
     bounds, baseline, p, gamma) = args
    
    delta = p / (2 * (p - 1))
    evaluation_count = 0
    max_evaluations = (len(parameter_names) + 1) * 6

    print(f"[{metric}] starting screening", flush=True)
    
    def fit_and_measure(params: dict[str, float]) -> dict[str, float]:
        """Fit model and return all 4 metrics."""
        nonlocal evaluation_count
        evaluation_count += 1
        print(
            f"[{metric}] model evaluation {evaluation_count}/{max_evaluations}",
            flush=True,
        )
        total_iter = int(round(params["iter"]))
        burn_in = int(round(params["burnin"] * total_iter))
        print(
            f"[{metric}] parameters: "
            f"m={int(round(params['m']))}, "
            f"nu={params['nu']:.6g}, "
            f"q={params['q']:.6g}, "
            f"omega={params['omega']:.6g}, "
            f"lambda_c={params['lambda_c']:.6g}, "
            f"iter={total_iter}, "
            f"burnin={params['burnin']:.6g}, "
            f"total_mcmc_iter={total_iter}, burn_in={burn_in}",
            flush=True,
        )
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
        
        # Measure fit time
        start_fit = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_fit
        
        # Measure predict time and get predictions
        start_pred = time.time()
        preds_train = model.predict(X_train)
        preds_test = model.predict(X_test)
        pred_time = time.time() - start_pred
        
        # Compute in-sample and out-of-sample RMSE
        rmse_in = float(np.sqrt(np.mean((y_train - preds_train) ** 2)))
        rmse_out = float(np.sqrt(np.mean((y_test - preds_test) ** 2)))
        
        return {
            "in_sample_rmse": rmse_in,
            "out_sample_rmse": rmse_out,
            "fit_time": fit_time,
            "pred_time": pred_time,
        }
    
    def objective_for_metric(params: dict[str, float]) -> float:
        measurements = fit_and_measure(params)
        return measurements[metric]

    def report_progress(record) -> None:
        names = [parameter_names[i] for i in record.eliminated_this_step]
        eliminated = ", ".join(names) if names else "none"
        print(
            f"[{metric}] iteration {record.R}: "
            f"{len(record.active_factors)} active, eliminated {eliminated}",
            flush=True,
        )

    simulate = make_parameter_simulator(
        objective_for_metric,
        parameter_names=parameter_names,
        bounds=bounds,
        baseline=baseline,
    )

    screen_start = time.time()
    result = sequential_ee_screen(
        simulate,
        k=len(parameter_names),
        m=6,
        delta=delta,
        gamma=gamma,
        seed=1,
        factor_names=parameter_names,
        progress_callback=report_progress,
    )
    screen_time = time.time() - screen_start
    
    return {
        "metric": metric,
        "result": result,
        "screen_time": screen_time,
    }


def _screen_one_metric_safe(args: tuple) -> tuple:
    """Wrapper around screen_one_metric that catches and returns exceptions.
    
    Returns: (metric, result_dict_or_None, error_or_None)
    """
    metric = args[0]
    try:
        return metric, screen_one_metric(args), None
    except Exception as exc:
        return metric, None, repr(exc)


def generate_plots_for_metric(args: tuple) -> dict:
    """Generate and save plots for one metric in a worker process.
    
    Args:
        args: (metric, result, root)
    
    Returns:
        Dictionary with metric and save paths.
    """
    metric, result, root = args
    
    out_dir = root / "outputs" / metric
    out_dir.mkdir(parents=True, exist_ok=True)
    
    ax1 = plot_final_effects(result)
    fig1 = ax1.figure
    
    fig2, axes2 = plot_iteration_grid(result)
    
    effects_path = out_dir / "final_effects.png"
    grid_path = out_dir / "iteration_grid.png"
    
    fig1.savefig(effects_path, dpi=300, bbox_inches="tight")
    fig2.savefig(grid_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    
    return {
        "metric": metric,
        "effects_path": str(effects_path),
        "grid_path": str(grid_path),
    }


def _generate_plots_safe(args: tuple) -> tuple:
    """Wrapper around generate_plots_for_metric that catches exceptions.
    
    Returns: (metric, plots_dict_or_None, error_or_None)
    """
    metric = args[0]
    try:
        return metric, generate_plots_for_metric(args), None
    except Exception as exc:
        return metric, None, repr(exc)


def default_n_workers() -> int:
    """Picks a worker count appropriate to how the job was launched.
    
    Respects SLURM's allocation if present, otherwise falls back to all
    visible cores, so the same script works both under sbatch/srun and when
    run manually.
    """
    for var in ("SLURM_CPUS_PER_TASK", "SLURM_NTASKS", "SLURM_CPUS_ON_NODE"):
        val = os.environ.get(var)
        if val:
            try:
                return int(val)
            except ValueError:
                pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def main(n_workers: int | None = None) -> None:
    """Screen AddiVortes hyperparameters across 4 metrics in parallel."""
    
    if n_workers is None:
        n_workers = default_n_workers()
    
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "benchmarks" / "datasets" / "boston"

    print("Loading data...")
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
    bounds["iter"] = (bounds["iter"][0], 5_000.0)

    print("Parameter ranges:")
    for name in parameter_names:
        print(f"  {name}: {bounds[name][0]} to {bounds[name][1]}")
    
    baseline = {
        "m": 200.0,
        "nu": 6.0,
        "q": 0.85,
        "omega": 3.0,
        "lambda_c": 25.0,
        "iter": 1200.0,
        "burnin": 0.1666666667,
    }

    metrics = ["in_sample_rmse", "out_sample_rmse", "fit_time", "pred_time"]
    p = 10
    gamma = 0.01
    
    screening_dir = Path(__file__).resolve().parent

    # Prepare arguments for worker processes
    # Each worker gets: (metric, X_train, y_train, X_test, y_test, 
    #                    parameter_names, bounds, baseline, p, gamma)
    worker_args = [
        (metric, X_train, y_train, X_test, y_test, parameter_names, 
         bounds, baseline, p, gamma)
        for metric in metrics
    ]

    # Screen all metrics in parallel
    n_workers = min(n_workers, len(metrics)) or 1
    print(f"\nScreening {len(metrics)} metrics across {n_workers} worker processes...")
    print(f"{'='*70}\n")

    metric_results = {}
    
    with mp.Pool(processes=n_workers, initializer=_init_worker) as pool:
        results_iter = pool.imap_unordered(
            _screen_one_metric_safe,
            worker_args,
            chunksize=1,
        )
        
        for i, (metric, result_dict, error) in enumerate(results_iter, 1):
            if error is not None:
                print(f"[FAILED] {metric} -> {error}\n")
                continue
            
            metric_results[metric] = result_dict["result"]
            screen_time = result_dict["screen_time"]
            print(f"[{i}/{len(metrics)}] Screening {metric} completed in {screen_time:.1f}s")
            print(result_dict["result"].summary())
            print()

    # Generate plots in parallel
    print(f"\nGenerating plots for all metrics...")
    print(f"{'='*70}\n")
    
    plot_args = [
        (metric, metric_results[metric], screening_dir)
        for metric in metrics
        if metric in metric_results
    ]

    plot_results = {}
    
    with mp.Pool(processes=n_workers, initializer=_init_worker) as pool:
        plots_iter = pool.imap_unordered(
            _generate_plots_safe,
            plot_args,
            chunksize=1,
        )
        
        for i, (metric, plots_dict, error) in enumerate(plots_iter, 1):
            if error is not None:
                print(f"[FAILED] Plotting {metric} -> {error}\n")
                continue
            
            plot_results[metric] = plots_dict
            print(f"[{i}/{len(plot_args)}] Plots saved for {metric}:")
            print(f"  - {plots_dict['effects_path']}")
            print(f"  - {plots_dict['grid_path']}\n")

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY: Parameter Importance Across Metrics")
    print(f"{'='*70}")
    
    for metric in metrics:
        if metric not in metric_results:
            print(f"\n{metric}: [SKIPPED - screening failed]")
            continue
        
        result = metric_results[metric]
        nonlinear = [parameter_names[i] for i in result.nonlinear_factors]
        linear = [parameter_names[i] for i in result.linear_factors]
        negligible = [parameter_names[i] for i in result.negligible_factors]
        
        print(f"\n{metric}:")
        print(f"  Nonlinear/important: {nonlinear}")
        print(f"  Linear: {linear}")
        print(f"  Negligible: {negligible}")
    
    print(f"\n{'='*70}")
    print("All screening complete!")


if __name__ == "__main__":
    main()
