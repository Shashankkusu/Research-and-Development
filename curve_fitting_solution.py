import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
# Fix charmap encoding error in Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    

def setup_matplotlib_for_plotting():
    """
    Setup matplotlib and seaborn for plotting with proper configuration.
    Call this function before creating any plots to ensure proper rendering.
    """
    import warnings
    import matplotlib.pyplot as plt
    
    # Ensure warnings are printed
    warnings.filterwarnings('default')  # Show all warnings
    
    # Configure matplotlib for non-interactive mode
    plt.switch_backend("Agg")
    
    # Configure platform-appropriate fonts for cross-platform compatibility
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "PingFang SC", "Arial Unicode MS", "Hiragino Sans GB"]
    plt.rcParams["axes.unicode_minus"] = False

def parametric_equations(t, theta, M, X):
    """
    Calculate x, y coordinates using the parametric equations
    """
    # Convert theta from degrees to radians if needed
    theta_rad = np.radians(theta)
    
    # Calculate x and y using the given parametric equations
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

def fit_curve_to_data(csv_file_path):
    """
    Fit the parametric curve to the observed data
    """
    # Load the data
    data = pd.read_csv(csv_file_path)
    x_obs = data['x'].values
    y_obs = data['y'].values
    
    # Since we need to find t values, we'll assume t is uniformly distributed
    # or we can try to estimate t from the data pattern
    # Let's start with uniform distribution in the range [6, 60]
    n_points = len(x_obs)
    t_data = np.linspace(6, 60, n_points)
    
    # Initial guess for parameters
    # theta in degrees, M, X
    initial_guess = [25.0, 0.01, 50.0]  # middle values of the ranges
    
    # Define bounds for the parameters
    bounds = [
        (0, 50),     # theta: 0 < theta < 50 degrees
        (-0.05, 0.05),  # M: -0.05 < M < 0.05
        (0, 100)     # X: 0 < X < 100
    ]
    
    print("Starting optimization...")
    print(f"Initial guess: theta={initial_guess[0]:.4f}, M={initial_guess[1]:.4f}, X={initial_guess[2]:.4f}")
    
    # Optimize
    result = minimize(
        objective_function, 
        initial_guess, 
        args=(t_data, x_obs, y_obs),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000}
    )
    
    if result.success:
        theta_opt, M_opt, X_opt = result.x
        print(f"Optimization successful!")
        print(f"Optimal parameters:")
        print(f"  theta = {theta_opt:.6f} degrees")
        print(f"  M = {M_opt:.6f}")
        print(f"  X = {X_opt:.6f}")
        print(f"  Final L1 distance: {result.fun:.6f}")
        
        return theta_opt, M_opt, X_opt, result.fun, t_data
    else:
        print(f"Optimization failed: {result.message}")
        return None, None, None, float('inf'), None

def plot_results(t_data, x_obs, y_obs, theta_opt, M_opt, X_opt):
    """
    Plot the original data and the fitted curve
    """
    setup_matplotlib_for_plotting()
    
    # Calculate predicted curve
    x_pred, y_pred = parametric_equations(t_data, theta_opt, M_opt, X_opt)
    
    plt.figure(figsize=(12, 8))
    
    # Plot observed data
    plt.scatter(x_obs, y_obs, alpha=0.6, color='blue', s=20, label='Observed Data')
    
    # Plot fitted curve
    plt.plot(x_pred, y_pred, 'r-', linewidth=2, label=f'Fitted Curve (θ={theta_opt:.4f}, M={M_opt:.4f}, X={X_opt:.4f})')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Curve Fitting Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('curve_fitting_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Results plot saved as 'curve_fitting_results.png'")

def analyze_t_estimation(csv_file_path):
    """
    Try to estimate better t values from the data pattern
    """
    data = pd.read_csv(csv_file_path)
    x_obs = data['x'].values
    y_obs = data['y'].values
    
    # Let's try different t estimation strategies
    
    # Strategy 1: Assume t is proportional to distance from origin or some center
    # First, estimate the center from the data
    x_center = np.mean(x_obs)
    y_center = np.mean(y_obs) - 42  # Subtract the known y offset
    
    # Calculate distance from estimated center
    distances = np.sqrt((x_obs - x_center)**2 + (y_obs - 42 - y_center)**2)
    
    # Strategy 2: Use the pattern in the data to estimate t
    # Since t appears linearly in the main terms, we can estimate t from the overall scale
    
    return t_data

def main():
    print("=== Parametric Curve Fitting Analysis ===\n")
    
    # Load and analyze the data
    csv_file = 'xy_data.csv'
    
    # First, let's examine the data
    data = pd.read_csv(csv_file)
    print(f"Data shape: {data.shape}")
    print(f"X range: [{data['x'].min():.3f}, {data['x'].max():.3f}]")
    print(f"Y range: [{data['y'].min():.3f}, {data['y'].max():.3f}]")
    print()
    
    # Try the curve fitting
    theta_opt, M_opt, X_opt, final_distance, t_data = fit_curve_to_data(csv_file)
    
    if theta_opt is not None:
        print(f"\n=== FINAL RESULTS ===")
        print(f"θ = {theta_opt:.6f} degrees")
        print(f"M = {M_opt:.6f}")
        print(f"X = {X_opt:.6f}")
        
        # Plot the results
        data = pd.read_csv(csv_file)
        x_obs = data['x'].values
        y_obs = data['y'].values
        plot_results(t_data, x_obs, y_obs, theta_opt, M_opt, X_opt)
        
        # Calculate and display the parameter values in LaTeX format
        print(f"\n=== LATEX FORMAT ===")
        print(f"\\left(t*\\cos({theta_opt:.6f})-e^{{{M_opt:.4f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({theta_opt:.6f})\\ +{X_opt:.4f},{42}+\\ t*\\sin({theta_opt:.6f})+e^{{{M_opt:.4f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({theta_opt:.6f})\\right)")
        
    else:
        print("Curve fitting failed!")

if __name__ == "__main__":
    main()