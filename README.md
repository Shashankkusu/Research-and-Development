# Parametric Curve — Parameter Estimation (Work Summary)

This repository contains the work done to estimate three unknown parameters (θ, M, X) of a parametric curve from measured (x, y) points for t in (6, 60). This is a practical implementation and documentation of the steps taken — a project/work report rather than a research paper.

Project snapshot
- Repository: https://github.com/Shashankkusu/Research-and-Development
- Description: Flam assesment
- Language: Python

## Overview

Problem (parametric curve)
Given:
x(t) = t + cos(θ) - e^{t M} * sin(0.3 t) * sin(θ) + X  
y(t) = 42 + t + sin(θ) + e^{t M} * sin(0.3 t) * cos(θ)

Unknowns to estimate from measured points:
- θ (theta): 0° < θ < 50°  (converted to radians in code)
- M: -0.05 < M < 0.05
- X: 0 < X < 100

Measured data: data/xy_data.csv (contains measured points; may include t or only x,y)

This README documents what was done, how to reproduce, evaluation metrics, outputs, and a short literature survey.

## What I did (summary of the work)
- Ingested the provided CSV of measured points and inspected whether t values are included.
- Implemented the forward parametric model x(t; θ,M,X) and y(t; θ,M,X).
- Implemented parameter estimation routines:
  - If t values are present: nonlinear least squares to fit θ, M, X.
  - If t values are not present: projection/assignment strategies or joint optimization that treats t_i as latent variables (bundle-adjustment style).
- Used SciPy optimization (local least_squares + optional global search via differential_evolution) and applied bounds on parameters.
- Produced plots showing measured points vs fitted curve and residual diagnostics.
- Saved final estimates and metrics to results/estimates.json and figures to results/plots/.

## Files in this repository (work layout)
- README.md — this file (project summary and reproduction steps)
- data/xy_data.csv — measured points (raw)
- notebooks/param_estimation.ipynb — exploratory notebook with code, plots, diagnostics
- scripts/estimate_params.py — command-line estimation script
- results/estimates.json — final parameter estimates and summary metrics (output)
- results/plots/ — saved figures (PNG)
- LICENSE — MIT license 


If any listed file is missing, see "How to reproduce" to generate results or create the skeleton files.

## Methods and implementation notes

Model functions
- Implemented x(t; θ,M,X) and y(t; θ,M,X) in Python, with θ converted to radians internally.

Estimation strategies
- Known t_i: minimize sum of squared residuals between measured and model coordinates using scipy.optimize.least_squares (supports bounds).
- Unknown t_i: two practical options implemented
  - Projection / grid-search alignment: sample t on a dense grid and map measured points to nearest model points (fast, practical).
  - Joint optimization: augment parameters with t_i and run nonlinear least squares with bounds for t_i (more accurate but higher dimensional).
- Global/local hybrid: differential_evolution for global exploration followed by local refinement with least_squares to reduce risk of poor local minima.

Parameter bounds used
- θ: (0, 50°) → (0, 50 * π / 180) radians
- M: (-0.05, 0.05)
- X: (0, 100)

Metrics and diagnostics
- For each measured point i, residual r_i = sqrt((x_i - x_model(t_i))^2 + (y_i - y_model(t_i))^2)
- Reported metrics: MAE, RMSE, and an L1 sampling-based metric used in the assessment (distance between uniformly-sampled predicted and expected curves)
- Visual diagnostics: fitted curve vs measured points, residuals vs t (or index), and residual histogram.

Numerical notes
- The term exp(t*M) stays in a numerically manageable range given constraints (exp(60*0.05) ~ 20). Monitor growth if bounds change.

## How to reproduce (quick start)

1. Clone
   git clone https://github.com/Shashankkusu/Research-and-Development.git
   cd Research-and-Development

2. Setup environment
   python -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1

3. Install dependencies
   pip install -r requirements.txt
   #  Requirements:
   pip install numpy scipy pandas matplotlib seaborn jupyter

4. Run parameter estimation script (example)
   python scripts/estimate_params.py --data data/xy_data.csv --out results/estimates.json

   Script options:
   - --method {local,global}  choose optimization strategy (default: local)
   - --n-samples N            number of t samples for projection-based approaches
   - --seed S                 random seed for reproducibility

5. Open notebook
   jupyter lab notebooks/param_estimation.ipynb

6. Inspect results
   - results/estimates.json contains final estimates and metrics
   - results/plots/fitted_curve.png and residuals plots show diagnostics

## Expected outputs (examples)
- results/estimates.json
  {
    "theta_rad": <value>,
    "theta_deg": <value>,
    "M": <value>,
    "X": <value>,
    "mae": <value>,
    "rmse": <value>,
    "L1_metric": <value>,
    "optimizer_info": { ... }
  }
- results/plots/fitted_curve.png
- results/plots/residuals.png

## Literature survey

The table below lists concise, practical references and short notes for methods and tools used. This is a lightweight literature survey provided to show the provenance of algorithms and libraries rather than to present new research.

| Reference | Type | Short summary | Relevance to this work |
|---|---:|---|---|
| SciPy: scipy.optimize.least_squares | Library docs | Implementation of Levenberg–Marquardt / Trust-region reflective least squares. | Used for local nonlinear least-squares parameter fitting. |
| Storn, R., & Price, K. (1997). Differential Evolution — A simple and efficient heuristic for global optimization | Algorithm paper (classic) | Differential evolution (DE) is a population-based global optimization method. | Used as an optional global search to avoid local minima before local refinement. |
| Nocedal, J., & Wright, S. (2006). Numerical Optimization. Springer. | Book | Comprehensive treatment of numerical optimization methods. | Practical reference for optimization strategies, constraints, and algorithm selection. |
| Triggs, B., McLauchlan, P. F., Hartley, R. I., & Fitzgibbon, A. W. (2000). Bundle Adjustment — A Modern Synthesis | Survey / paper | Describes joint optimization of parameters and latent variables (e.g., camera poses and 3D points). | Conceptually similar to treating t_i as latent variables when t is not measured. |
| Wikipedia / online notes — Levenberg–Marquardt | Online reference | Practical description of LM algorithm and typical use in curve-fitting. | Helps interpret solver options used in SciPy and tuning of jacobian/weights. |

Notes:
- These entries are practical references; this repository is a work/project implementation and the above is provided for context and reproducibility, not for claiming novel research.

## Notes / scope
- This is a project/work assignment implementation and documentation; it is not a research paper.
- The repository documents choices made, scripts used, and outputs so others can reproduce the steps.
- If you later want to expand into a research-style report, the literature survey can be extended and formal citations added in docs/report.pdf.



## License
- MIT License.

## Contact
- Repository owner: @Shashankkusu
- For questions or issues: open an issue in this repository.

---

