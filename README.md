# Parametric Curve Fitting — README

Author: Shashankkusu  
Updated: 2025-11-09

This repository contains a single Python script used to fit a parametric 2D curve to observed (x,y) data points in `xy_data.csv`.

Files
- `curve_fitting_solution.py` — The single Python script used for the analysis and plotting.
- `xy_data.csv` — Input dataset (expected: two columns `x` and `y`). Replace with your own data to re-run.

Summary of what this script does
- Implements the parametric model
  - x(t) = t*cos(θ) - exp(M * |t|) * sin(0.3 t) * sin(θ) + X
  - y(t) = 42 + t*sin(θ) + exp(M * |t|) * sin(0.3 t) * cos(θ)
- Uses a single-stage optimization that minimizes the L1 distance (sum of absolute residuals in x and y) between observed points and predicted points.
- Uses a fixed t-array (uniformly spaced between 6 and 60) with the same number of t values as data points.
- Optimizes three parameters: θ (in degrees, the code expects/prints degrees), M (exponential coefficient), and X (horizontal shift).
- Produces a PNG plot of observed points and the fitted curve and prints final parameter values and a LaTeX-friendly expression to stdout.

How the current implementation works (key details)
- t values: uniform linspace from 6 to 60 (one t per observed point).
- Objective: L1 distance computed as sum(|x_obs - x_pred| + |y_obs - y_pred|).
- Optimization: scipy.optimize.minimize with L-BFGS-B and bounds.
- Initial guess (in the script):
  - theta = 25.0 (degrees)
  - M = 0.01
  - X = 50.0
- Parameter bounds (as implemented):
  - theta: [0, 50] degrees
  - M: [-0.05, 0.05]
  - X: [0, 100]
- Plot output: `curve_fitting_results.png`
- LaTeX output: printed to console (a single-line parametric expression).

Dependencies
- Python 3 (3.8+ recommended)
- numpy
- pandas
- scipy
- matplotlib

Install dependencies:
```bash
pip install numpy pandas scipy matplotlib
```

Usage
1. Place your `xy_data.csv` in the same directory as `curve_fitting_solution.py`. The script expects two columns named `x` and `y`.
2. Run:
```bash
python curve_fitting_solution.py
```
3. Outputs:
   - Console: optimization progress and final parameter values (θ in degrees, M, X) and a LaTeX-like expression.
   - File: `curve_fitting_results.png` — scatter of observed data and the fitted curve.

Notes, limitations and suggestions
- Current design uses a fixed t ordering (uniform t). If true per-point t correspondences are unknown, the fit may be suboptimal. A more robust L1 two-level approach (inner optimization to find best t per point) is not implemented in this script.
- The script treats θ in degrees internally (conversion to radians is performed inside parametric equations). If you prefer to use radians everywhere, change input/printing accordingly.
- Bound choices (especially for M) and the initial guess can strongly affect the result due to the exponential term exp(M|t|). If you see unstable results, tighten bounds for M or restrict t range.
- If you need robustness to outliers with adaptive t per point (recommended for noisy datasets), consider enhancing the script:
  - Implement per-point inner minimization to find argmin_t distance to each observed point (bounded t ∈ [6,60]).
  - Use the outer optimizer to minimize the sum of per-point L1 distances computed using those inner-optimal t values.
  - Parallelize inner solves for speed when dataset is larger.

Example LaTeX output format (printed by the script)
\left(t*\cos(θ)-e^{M\left|t\right|}\cdot\sin(0.3t)\sin(θ)+X, 42 + t*\sin(θ)+e^{M\left|t\right|}\cdot\sin(0.3t)\cos(θ)\right)

What I changed / why this README is updated
- You indicated you only used the Python code `curve_fitting_solution.py`. This README was updated to accurately describe what exists in the repository (only that script and the dataset), the algorithm it implements, how to run it, its outputs, limitations, and recommended next steps for more robust fitting.

If you'd like, I can:
- Add a short example of expected console output after a successful run.
- Adjust the script to perform two-level L1 fitting (inner per-point t optimization + outer parameter optimization).
- Convert θ handling to radians end-to-end, or add CLI flags for parameters and bounds.

```
