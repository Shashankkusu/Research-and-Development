import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, basinhopping
import matplotlib.pyplot as plt
import sys
# Fix charmap encoding error in Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def setup_matplotlib_for_plotting():
    """
    Setup matplotlib for plotting with proper configuration.
    """
    import warnings
    import matplotlib.pyplot as plt

    warnings.filterwarnings('default')
    plt.switch_backend("Agg")

    # Use widely available fonts
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Verdana"]
    plt.rcParams["axes.unicode_minus"] = False


def parametric_equations(t, theta, M, X):
    """
    Calculate x, y coordinates using the parametric equations
    """
    theta_rad = np.radians(theta)
    
    x = t * np.cos(theta_rad) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta_rad) + X
    y = 42 + t * np.sin(theta_rad) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta_rad)
    
    return x, y

def objective_function(params, t_data, x_obs, y_obs):
    """
    Objective function to minimize - L1 distance between observed and predicted points
    """
    theta, M, X = params
    
    # Calculate predicted coordinates
    x_pred, y_pred = parametric_equations(t_data, theta, M, X)
    
    # Calculate L1 distance
    l1_distance = np.sum(np.abs(x_obs - x_pred) + np.abs(y_obs - y_pred))
    
    return l1_distance

def estimate_t_by_radius(x_obs, y_obs):
    """
    Estimate t values by sorting points by distance from center
    """
    n_points = len(x_obs)
    
    # Estimate center
    x_center = np.mean(x_obs)
    y_center = np.mean(y_obs) - 42  # Subtract known y offset
    
    # Calculate distances from center
    distances = np.sqrt((x_obs - x_center)**2 + (y_obs - y_center)**2)
    sorted_indices = np.argsort(distances)
    
    # Create t values
    t_values = np.linspace(6, 60, n_points)
    
    # Map back to original order
    t_reordered = np.zeros(n_points)
    t_reordered[sorted_indices] = t_values
    
    return t_reordered

def refined_optimization(csv_file_path):
    """
    Perform refined optimization using the best strategy
    """
    data = pd.read_csv(csv_file_path)
    x_obs = data['x'].values
    y_obs = data['y'].values
    
    # Use the best t estimation method
    t_data = estimate_t_by_radius(x_obs, y_obs)
    
    print("=== Refined Optimization ===\\n")
    
    bounds = [
        (0, 50),     # theta: 0 < theta < 50 degrees
        (-0.05, 0.05),  # M: -0.05 < M < 0.05
        (0, 100)     # X: 0 < X < 100
    ]
    
    best_result = None
    best_distance = float('inf')
    
    # Multiple optimization strategies
    strategies = [
        # Differential Evolution (global optimization)
        {
            'method': 'differential_evolution',
            'kwargs': {
                'maxiter': 2000,
                'popsize': 20,
                'seed': 42,
                ' atol': 1e-6,
                'tol': 1e-6
            }
        },
        
        # Basinhopping (global optimization with local search)
        {
            'method': 'basinhopping',
            'kwargs': {
                'niter': 100,
                'T': 1.0,
                'stepsize': 2.0,
                'minimizer_kwargs': {
                    'method': 'L-BFGS-B',
                    'bounds': bounds
                }
            },
            'x0': [30, 0.03, 55]
        },
        
        # Multiple L-BFGS-B starts
        {
            'method': 'L-BFGS-B_multi',
            'starts': [
                [30, 0.03, 55],
                [25, 0.02, 50],
                [35, 0.04, 60],
                [28, 0.025, 52],
                [32, 0.035, 58]
            ]
        }
    ]
    
    for i, strategy in enumerate(strategies):
        print(f"Strategy {i+1}: {strategy['method']}")
        
        if strategy['method'] == 'differential_evolution':
            try:
                result = differential_evolution(
                    objective_function,
                    bounds,
                    args=(t_data, x_obs, y_obs),
                    **strategy['kwargs']
                )
                
                if result.success and result.fun < best_distance:
                    best_result = result
                    best_distance = result.fun
                    print(f"  Success: L1 = {result.fun:.6f}")
                else:
                    print(f"  Failed or worse: L1 = {result.fun if result.success else 'failed'}")
                    
            except Exception as e:
                print(f"  Exception: {e}")
        
        elif strategy['method'] == 'basinhopping':
            try:
                result = basinhopping(
                    objective_function,
                    strategy['x0'],
                    args=(t_data, x_obs, y_obs),
                    **strategy['kwargs']
                )
                
                if result.success and result.fun < best_distance:
                    best_result = result
                    best_distance = result.fun
                    print(f"  Success: L1 = {result.fun:.6f}")
                else:
                    print(f"  Failed or worse: L1 = {result.fun if result.success else 'failed'}")
                    
            except Exception as e:
                print(f"  Exception: {e}")
        
        elif strategy['method'] == 'L-BFGS-B_multi':
            for j, start in enumerate(strategy['starts']):
                try:
                    result = minimize(
                        objective_function,
                        start,
                        args=(t_data, x_obs, y_obs),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 2000, 'ftol': 1e-9}
                    )
                    
                    if result.success and result.fun < best_distance:
                        best_result = result
                        best_distance = result.fun
                        print(f"  Start {j+1} success: L1 = {result.fun:.6f}")
                    else:
                        print(f"  Start {j+1}: L1 = {result.fun if result.success else 'failed'}")
                        
                except Exception as e:
                    print(f"  Start {j+1} exception: {e}")
        
        print()
    
    return best_result, t_data

def plot_final_results(t_data, x_obs, y_obs, theta_opt, M_opt, X_opt, distance):
    """
    Plot the final fitting results
    """
    setup_matplotlib_for_plotting()
    
    # Calculate predicted curve
    x_pred, y_pred = parametric_equations(t_data, theta_opt, M_opt, X_opt)
    
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Overall view
    plt.subplot(1, 3, 1)
    plt.scatter(x_obs, y_obs, alpha=0.6, color='blue', s=20, label='Observed Data')
    plt.plot(x_pred, y_pred, 'r-', linewidth=2, label=f'Fitted Curve')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Overall View\\nθ={theta_opt:.3f}°, M={M_opt:.4f}, X={X_opt:.3f}\\nL1={distance:.1f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Subplot 2: X coordinates comparison
    plt.subplot(1, 3, 2)
    plt.scatter(t_data, x_obs, alpha=0.6, color='blue', s=10, label='Observed X')
    plt.plot(t_data, x_pred, 'r-', linewidth=2, label='Predicted X')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('X vs t')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Y coordinates comparison
    plt.subplot(1, 3, 3)
    plt.scatter(t_data, y_obs, alpha=0.6, color='blue', s=10, label='Observed Y')
    plt.plot(t_data, y_pred, 'r-', linewidth=2, label='Predicted Y')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Y vs t')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Final results plot saved as 'final_results.png'")

def analyze_fit_quality(x_obs, y_obs, x_pred, y_pred, t_data):
    """
    Analyze the quality of the fit
    """
    print("=== Fit Quality Analysis ===")
    
    # Calculate various error metrics
    l1_error = np.sum(np.abs(x_obs - x_pred) + np.abs(y_obs - y_pred))
    l2_error = np.sqrt(np.sum((x_obs - x_pred)**2 + (y_obs - y_pred)**2))
    
    # Mean absolute error
    mae_x = np.mean(np.abs(x_obs - x_pred))
    mae_y = np.mean(np.abs(y_obs - y_pred))
    
    # Root mean square error
    rmse_x = np.sqrt(np.mean((x_obs - x_pred)**2))
    rmse_y = np.sqrt(np.mean((y_obs - y_pred)**2))
    
    print(f"L1 Error: {l1_error:.6f}")
    print(f"L2 Error: {l2_error:.6f}")
    print(f"MAE X: {mae_x:.6f}")
    print(f"MAE Y: {mae_y:.6f}")
    print(f"RMSE X: {rmse_x:.6f}")
    print(f"RMSE Y: {rmse_y:.6f}")
    
    # Point-by-point errors
    point_errors = np.abs(x_obs - x_pred) + np.abs(y_obs - y_pred)
    print(f"Max point error: {np.max(point_errors):.6f}")
    print(f"Min point error: {np.min(point_errors):.6f}")
    print(f"Median point error: {np.median(point_errors):.6f}")
    
    return {
        'l1': l1_error,
        'l2': l2_error,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'max_error': np.max(point_errors),
        'min_error': np.min(point_errors),
        'median_error': np.median(point_errors)
    }

def main():
    print("=== Final Refined Optimization ===\\n")
    
    csv_file = 'xy_data.csv'
    
    # Load data
    data = pd.read_csv(csv_file)
    x_obs = data['x'].values
    y_obs = data['y'].values
    
    print(f"Data: {len(x_obs)} points")
    print(f"X range: [{x_obs.min():.3f}, {x_obs.max():.3f}]")
    print(f"Y range: [{y_obs.min():.3f}, {y_obs.max():.3f}]")
    print()
    
    # Run refined optimization
    best_result, t_data = refined_optimization(csv_file)
    
    if best_result is not None and best_result.success:
        theta_opt, M_opt, X_opt = best_result.x
        
        print(f"=== FINAL OPTIMIZED RESULTS ===")
        print(f"θ = {theta_opt:.6f} degrees")
        print(f"M = {M_opt:.6f}")
        print(f"X = {X_opt:.6f}")
        print(f"L1 Distance = {best_result.fun:.6f}")
        
        # Calculate predicted curve
        x_pred, y_pred = parametric_equations(t_data, theta_opt, M_opt, X_opt)
        
        # Analyze fit quality
        quality_metrics = analyze_fit_quality(x_obs, y_obs, x_pred, y_pred, t_data)
        
        # Plot results
        plot_final_results(t_data, x_obs, y_obs, theta_opt, M_opt, X_opt, best_result.fun)
        
        # LaTeX format
        print(f"\\n=== LATEX FORMAT ===")
        latex_result = f"\\\\left(t*\\\\cos({theta_opt:.6f})-e^{{{M_opt:.4f}\\\\left|t\\\\right|}}\\\\cdot\\\\sin(0.3t)\\\\sin({theta_opt:.6f})\\\\ +{X_opt:.4f},{42}+\\\\ t*\\\\sin({theta_opt:.6f})+e^{{{M_opt:.4f}\\\\left|t\\\\right|}}\\\\cdot\\\\sin(0.3t)\\\\cos({theta_opt:.6f})\\\\right)"
        print(latex_result)
        
        # Clean version for submission
        clean_latex = f"\\left(t*\\cos({theta_opt:.6f})-e^{{{M_opt:.4f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({theta_opt:.6f})\\ +{X_opt:.4f},{42}+\\ t*\\sin({theta_opt:.6f})+e^{{{M_opt:.4f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({theta_opt:.6f})\\right)"
        print(f"\\n=== CLEAN LATEX FOR SUBMISSION ===")
        print(clean_latex)
        
        # Save results to file
        with open('final_results.txt', 'w', encoding='utf-8') as f:
            f.write("=== FINAL OPTIMIZATION RESULTS ===\n\n")
            f.write(f"θ (Theta)        : {theta_opt:.6f} degrees\n")
            f.write(f"M (Exponent)     : {M_opt:.6f}\n")
            f.write(f"X (Shift)        : {X_opt:.6f}\n")
            f.write(f"L1 Distance      : {best_result.fun:.6f}\n")
            f.write("\n----------------------------------------\n")
            f.write("📘 LaTeX Equation:\n")
            f.write(f"{clean_latex}\n")
            f.write("----------------------------------------\n\n")
            f.write("📊 Quality Metrics\n")
            f.write("----------------------------------------\n")
            f.write(f"{'Metric':<15}{'Value'}\n")
            f.write(f"{'-'*25}\n")
            for key, value in quality_metrics.items():
                f.write(f"{key:<15}{value:.6f}\n")
            f.write("----------------------------------------\n")
            f.write("✅ Results saved successfully.\n")
        print(f"\\nResults saved to 'final_results.txt'")
        
    else:
        print("Optimization failed!")

if __name__ == "__main__":
    main()