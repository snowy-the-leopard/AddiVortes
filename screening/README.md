# Sequential Elementary Effects Screening for AddiVortes

Implementation of the sequential screening algorithm from:

> Boukouvalas, A., Gosling, J.P., and Maruri-Aguilar, H. (2014). *An Efficient
> Screening Method for Computer Experiments.* Technometrics, 56(4), 422-431.

This package is focused on screening a black-box scalar objective and the
AddiVortes parameters that govern it. The screened inputs are the parameters
themselves, not variables from a particular dataset.

## Files

- `sequential_ee.py` — core algorithm: maximin LHS design, farthest-point
  ordering, OAT elementary-effect trajectories, the sigma0 chi-squared
  threshold (Lemma 1 / Eq. 5), and Algorithm 1's sequential loop.
- `addivortes_wrapper.py` — maps a unit-cube point into a dictionary of
  AddiVortes parameter values using the ranges from parameters.tex.
- `plotting.py` — the mu*-vs-sigma effect plots from Figures 1 and 3 of the
  paper.
- `example_addivortes_parameter_screening.py` — a runnable example that
  screens a scalar objective over AddiVortes parameters.
- `tests/test_parameter_screening.py` — regression tests for the parameter
  wrapper and the LaTeX range parser.

## Validation against the paper

`sequential_ee.py` was checked directly against the paper's worked examples
before being wired up to AddiVortes:

- **Example 1** (farthest-point ordering of 6 points in 5D): the ordering
  produced (`x2, x5, x4, x6, x1, x3`) matches the paper exactly.
- **Section 4.1** sigma0 formula: for R=6, delta=5/9, sqrt(gamma)=0.087, the
  derived threshold is 0.3847, matching the paper's reported 0.385.
- **Example 2** (5-input test function): the algorithm correctly separates
  {x1, x2} as nonlinear from {x3, x4, x5} as linear/negligible, gets x4's
  elementary effect exactly zero (it has no effect on the function), and
  uses **28 simulator runs**, exactly matching the paper's reported total
  (vs. 36 for the batch method — a 22% saving, also matching).
- **Section 5** (20-input Morris test function): correctly separates
  x1-x7 (nonlinear), x8-x10 (linear), and the great majority of x11-x20
  (negligible), consistent with the paper's own reported ~92-99% accuracy
  (it isn't 100% in the paper either).

## Basic workflow

```python
from screening.addivortes_wrapper import load_parameter_bounds_from_tex, make_parameter_simulator
from screening.sequential_ee import sequential_ee_screen


def objective(params):
    return (
        abs(params["m"] - 200) / 100
        + abs(params["nu"] - 6) / 10
        + abs(params["q"] - 0.85)
    )

bounds = load_parameter_bounds_from_tex("parameters.tex")
parameter_names = ["m", "nu", "q"]
simulate = make_parameter_simulator(
    objective,
    parameter_names=parameter_names,
    bounds={name: bounds[name] for name in parameter_names},
)

result = sequential_ee_screen(
    simulate, k=len(parameter_names), m=10, delta=0.556, gamma=0.01,
    seed=1, factor_names=parameter_names,
)
print(result.summary())
```

## Choosing the knobs

- **`p` / `delta`** — grid resolution for the OAT step. `delta = p/(2*(p-1))`
  is the paper's recommendation; p=10 (delta≈0.556) or p=20 (delta≈0.526)
  are reasonable defaults for k up to a few dozen inputs.
- **`gamma`** — the tolerated *output-scale* variance for "close enough to
  linear" (Eq. 3 in the paper). Think of it as: "I don't mind treating an
  input as linear if the response wiggles by no more than X around a
  straight line." If X is your tolerance and you want that to be roughly
  a 3-sigma band (as the paper does for the rabies example),
  `sqrt(gamma) = X/3`.
- **`m`** — space-filling design budget. Worst case cost is `(k+1)*m`
  simulator calls (same as running the batch method with R=m trajectories);
  actual cost will usually be well below that once some factors are
  eliminated early.

## Notes for parameter screening

- The parameter ranges are read from parameters.tex, so you can keep the
  screening space aligned with the project documentation.
- The objective should return a single scalar score for each parameter
  dictionary. That score can be validation RMSE, a surrogate fitness metric,
  or any other black-box quantity you want to study.
