# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt
import sys

sys.stdout.reconfigure(encoding='utf-8')
# Load the data
data = pd.read_csv('xy_data.csv')
x_data = data['x'].values
y_data = data['y'].values

print(f"Loaded {len(x_data)} data points")
print(f"X range: [{x_data.min():.2f}, {x_data.max():.2f}]")
print(f"Y range: [{y_data.min():.2f}, {y_data.max():.2f}]")

# Define the parametric equations
def parametric_curve(t, theta, M, X):
    x = t * np.cos(theta) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(theta)
    return x, y

# Smart approach: Since y = 42 + ... and y ranges from ~46 to ~70,
# we can use the y-42 relationship to estimate t initially

print("\nAnalyzing data structure...")

# The y equation is: y = 42 + t*sin(θ) + e^(M|t|)*sin(0.3t)*cos(θ)
# For small M, e^(M|t|) ≈ 1, and the oscillating term is small
# So y - 42 ≈ t*sin(θ) gives us a rough t estimate

# Similarly for x: x = t*cos(θ) - e^(M|t|)*sin(0.3t)*sin(θ) + X
# x - X ≈ t*cos(θ) (ignoring the oscillating term)

# Therefore: (y-42)/sin(θ) ≈ t and (x-X)/cos(θ) ≈ t
# This gives us: tan(θ) ≈ (y-42)/(x-X)

# Use grid search over theta to find best fit
print("\nGrid search over parameters...")

best_error = float('inf')
best_params = None

# Coarse grid search
theta_range = np.linspace(0.1, 0.8, 15)  # 0-46 degrees
M_range = np.linspace(-0.04, 0.04, 9)
X_range = np.linspace(10, 90, 17)

results = []

for theta in theta_range:
    for M in M_range:
        for X in X_range:
            # For each parameter set, estimate t values efficiently
            # Use vectorized computation with a simple t estimation
            
            # Rough t estimation from linear approximation
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            
            if abs(sin_theta) > 0.1:  # Avoid division by zero
                t_est = (y_data - 42) / sin_theta
            else:
                t_est = (x_data - X) / (cos_theta + 1e-10)
            
            # Clip to valid range
            t_est = np.clip(t_est, 6, 60)
            
            # Calculate error
            x_pred, y_pred = parametric_curve(t_est, theta, M, X)
            errors = np.abs(x_pred - x_data) + np.abs(y_pred - y_data)
            total_error = np.sum(errors)
            
            results.append((total_error, theta, M, X))
            
            if total_error < best_error:
                best_error = total_error
                best_params = (theta, M, X)

# Sort results
results.sort()

print(f"\nTop 10 parameter sets from grid search:")
for i in range(min(10, len(results))):
    err, th, m, x = results[i]
    print(f"  {i+1}. Error={err:.2f}, theta={th:.4f} ({np.degrees(th):.1f}°), M={m:.4f}, X={x:.2f}")

# Refine top candidates
print("\nRefining top candidates...")

def objective_refined(params):
    theta, M, X = params
    
    # Better t estimation
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    
    # Use both equations to estimate t
    if abs(sin_theta) > abs(cos_theta):
        t_est = (y_data - 42) / (sin_theta + 1e-10)
    else:
        t_est = (x_data - X) / (cos_theta + 1e-10)
    
    t_est = np.clip(t_est, 6, 60)
    
    # Refine t estimation with one Newton step
    for _ in range(2):
        x_pred, y_pred = parametric_curve(t_est, theta, M, X)
        
        # Compute approximate derivatives
        eps = 0.1
        x_pred_plus, y_pred_plus = parametric_curve(t_est + eps, theta, M, X)
        dx_dt = (x_pred_plus - x_pred) / eps
        dy_dt = (y_pred_plus - y_pred) / eps
        
        # Newton update for minimizing (x_pred-x_data)^2 + (y_pred-y_data)^2
        residual_x = x_pred - x_data
        residual_y = y_pred - y_data
        
        dt = -(residual_x * dx_dt + residual_y * dy_dt) / (dx_dt**2 + dy_dt**2 + 1e-10)
        t_est = np.clip(t_est + 0.5 * dt, 6, 60)
    
    # Final error
    x_pred, y_pred = parametric_curve(t_est, theta, M, X)
    errors = np.abs(x_pred - x_data) + np.abs(y_pred - y_data)
    return np.sum(errors)

# Refine top 5 candidates
top_refined = []
for i in range(min(5, len(results))):
    _, theta, M, X = results[i]
    
    print(f"\nRefining candidate {i+1}...")
    print(f"  Initial: theta={theta:.4f}, M={M:.4f}, X={X:.2f}")
    
    result = minimize(
        objective_refined,
        x0=[theta, M, X],
        method='L-BFGS-B',
        bounds=[(0, 0.873), (-0.05, 0.05), (0, 100)],
        options={'maxiter': 100}
    )
    
    print(f"  Refined: theta={result.x[0]:.6f} ({np.degrees(result.x[0]):.2f}°), M={result.x[1]:.6f}, X={result.x[2]:.4f}")
    print(f"  Error: {result.fun:.2f}")
    
    top_refined.append((result.fun, result.x))

# Select best
top_refined.sort()
best_error, best_params = top_refined[0]
theta_final, M_final, X_final = best_params

print("\n" + "="*70)
print("BEST SOLUTION FOUND")
print("="*70)
print(f"θ (theta) = {theta_final:.6f} radians = {np.degrees(theta_final):.4f} degrees")
print(f"M = {M_final:.6f}")
print(f"X = {X_final:.6f}")
print("="*70)

# Final evaluation with best t estimation
print("\nFinal evaluation on full dataset...")

sin_theta = np.sin(theta_final)
cos_theta = np.cos(theta_final)

# Initial t estimation
if abs(sin_theta) > abs(cos_theta):
    t_final = (y_data - 42) / (sin_theta + 1e-10)
else:
    t_final = (x_data - X_final) / (cos_theta + 1e-10)

t_final = np.clip(t_final, 6, 60)

# Refine with Newton iterations
for iteration in range(5):
    x_pred, y_pred = parametric_curve(t_final, theta_final, M_final, X_final)
    
    eps = 0.1
    x_pred_plus, y_pred_plus = parametric_curve(t_final + eps, theta_final, M_final, X_final)
    dx_dt = (x_pred_plus - x_pred) / eps
    dy_dt = (y_pred_plus - y_pred) / eps
    
    residual_x = x_pred - x_data
    residual_y = y_pred - y_data
    
    dt = -(residual_x * dx_dt + residual_y * dy_dt) / (dx_dt**2 + dy_dt**2 + 1e-10)
    t_final = np.clip(t_final + 0.5 * dt, 6, 60)

# Calculate final errors
x_pred, y_pred = parametric_curve(t_final, theta_final, M_final, X_final)

l1_errors = np.abs(x_pred - x_data) + np.abs(y_pred - y_data)
l2_errors = np.sqrt((x_pred - x_data)**2 + (y_pred - y_data)**2)

total_l1 = np.sum(l1_errors)
total_l2 = np.sum(l2_errors)

print(f"\nError Metrics:")
print(f"  Total L1 distance: {total_l1:.4f}")
print(f"  Average L1 per point: {total_l1/len(x_data):.6f}")
print(f"  Average L2 per point: {total_l2/len(x_data):.6f}")
print(f"  Median L2 error: {np.median(l2_errors):.6f}")
print(f"  Max L2 error: {np.max(l2_errors):.4f}")
print(f"  95th percentile: {np.percentile(l2_errors, 95):.4f}")

# Desmos format
print("\n" + "="*70)
print("DESMOS FORMAT:")
print("="*70)
desmos = f"\\left(t\\cdot\\cos({theta_final:.6f})-e^{{{M_final:.6f}\\cdot\\left|t\\right|}}\\cdot\\sin\\left(0.3t\\right)\\cdot\\sin({theta_final:.6f})+{X_final:.6f},42+t\\cdot\\sin({theta_final:.6f})+e^{{{M_final:.6f}\\cdot\\left|t\\right|}}\\cdot\\sin\\left(0.3t\\right)\\cdot\\cos({theta_final:.6f})\\right)"
print(desmos)
print("="*70)

# Save results
with open('solution.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("PARAMETRIC CURVE FITTING SOLUTION\n")
    f.write("="*70 + "\n\n")
    f.write("UNKNOWN VARIABLES:\n")
    f.write("-"*70 + "\n")
    f.write(f"θ = {theta_final:.6f} radians = {np.degrees(theta_final):.4f} degrees\n")
    f.write(f"M = {M_final:.6f}\n")
    f.write(f"X = {X_final:.6f}\n")
    f.write("-"*70 + "\n\n")
    f.write("Constraints Check:\n")
    f.write(f"  0° < θ < 50°     : {'✓' if 0 < np.degrees(theta_final) < 50 else '✗'}\n")
    f.write(f"  -0.05 < M < 0.05 : {'✓' if -0.05 < M_final < 0.05 else '✗'}\n")
    f.write(f"  0 < X < 100      : {'✓' if 0 < X_final < 100 else '✗'}\n\n")
    f.write("Error Metrics:\n")
    f.write(f"  Total L1 distance: {total_l1:.4f}\n")
    f.write(f"  Average L1 per point: {total_l1/len(x_data):.6f}\n")
    f.write(f"  Average L2 per point: {total_l2/len(x_data):.6f}\n\n")
    f.write("Desmos Format:\n")
    f.write("="*70 + "\n")
    f.write(desmos + "\n")
    f.write("="*70 + "\n")

# Visualization
fig = plt.figure(figsize=(15, 10))

ax1 = plt.subplot(2, 3, 1)
t_plot = np.linspace(6, 60, 1000)
x_plot, y_plot = parametric_curve(t_plot, theta_final, M_final, X_final)
ax1.plot(x_plot, y_plot, 'r-', linewidth=2.5, label='Fitted curve')
ax1.scatter(x_data, y_data, alpha=0.2, s=5, c='blue', label='Data')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Fitted Curve vs Data')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 3, 2)
ax2.hist(l2_errors, bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(np.mean(l2_errors), color='red', linestyle='--', label=f'Mean: {np.mean(l2_errors):.3f}')
ax2.set_xlabel('L2 Error')
ax2.set_ylabel('Frequency')
ax2.set_title('Error Distribution')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 3, 3)
ax3.hist(t_final, bins=50, edgecolor='black', alpha=0.7, color='green')
ax3.set_xlabel('t parameter')
ax3.set_ylabel('Frequency')
ax3.set_title('t Distribution')
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(2, 3, 4)
ax4.scatter(range(len(x_data)), x_pred - x_data, alpha=0.3, s=3, label='x residuals')
ax4.scatter(range(len(y_data)), y_pred - y_data, alpha=0.3, s=3, label='y residuals')
ax4.axhline(y=0, color='black', linestyle='--')
ax4.set_xlabel('Data point index')
ax4.set_ylabel('Residual')
ax4.set_title('Residuals')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = plt.subplot(2, 3, 5)
scatter = ax5.scatter(x_data, y_data, c=t_final, cmap='viridis', s=10, alpha=0.6)
plt.colorbar(scatter, ax=ax5, label='t')
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_title('Data Colored by t')
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(2, 3, 6)
ax6.scatter(t_final, l2_errors, alpha=0.3, s=10, c='orange')
ax6.set_xlabel('t')
ax6.set_ylabel('L2 Error')
ax6.set_title('Error vs t')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('solution_plots.png', dpi=150, bbox_inches='tight')

print("\nResults saved to solution.txt")
print("Plots saved to solution_plots.png")
print("\nDONE!")