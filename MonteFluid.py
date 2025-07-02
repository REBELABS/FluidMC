# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:31:37 2025

@author: HomePC
"""
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Parameters from literature review
def vs_af_model(theta):
    """VS model for Ascending Flow (AF)"""
    return -1.50e-6 * theta*3 + 1.76e-4 * theta*2 - 3.05e-3 * theta + 2.75e-2

def matrix_af_model(theta):
    """Matrix model for Ascending Flow (AF)"""
    return 6.04e-3 * theta - 1.04e-16

def vs_df_model(theta):
    """VS model for Descending Flow (DF)"""
    return -3.68e-7 * theta*3 + 5.11e-5 * theta*2 - 7.00e-3 * theta + 4.88e-1

# Generate synthetic data using Monte Carlo simulation
def generate_synthetic_data(n_samples=100000, theta_min=0, theta_max=1):
    """Generate synthetic theta values and corresponding q values"""
    theta = np.random.uniform(theta_min, theta_max, n_samples)
    q_vs_af = vs_af_model(theta)
    q_matrix_af = matrix_af_model(theta)
    q_vs_df = vs_df_model(theta)
    return theta, q_vs_af, q_matrix_af, q_vs_df
    print(theta)

# Error metrics calculation
def calculate_error_metrics(y_true, y_pred):
    """Calculate RMSE, MAE, and R²"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# Sensitivity analysis (One-At-A-Time)
def sensitivity_analysis(theta, q_pred, param_range):
    """Perform OAT sensitivity analysis"""
    sensitivity = []
    base_q = np.mean(q_pred)
    for val in param_range:
        theta_perturbed = theta * val
        q_perturbed = vs_af_model(theta_perturbed)  # Using VS AF model as example
        sensitivity.append(np.mean(np.abs(q_perturbed - base_q)))
    return sensitivity

# Main execution
if _name_ == "_main_":
    # Generate synthetic data
    n_samples = 100000
    theta, q_vs_af_true, q_matrix_af_true, q_vs_df_true = generate_synthetic_data(n_samples)

    # Add some noise to simulate real data
    noise = np.random.normal(0, 0.01, n_samples)
    q_vs_af_pred = vs_af_model(theta) + noise
    q_matrix_af_pred = matrix_af_model(theta) + noise
    q_vs_df_pred = vs_df_model(theta) + noise

    # Calculate error metrics
    rmse_vs_af, mae_vs_af, r2_vs_af = calculate_error_metrics(q_vs_af_true, q_vs_af_pred)
    rmse_matrix_af, mae_matrix_af, r2_matrix_af = calculate_error_metrics(q_matrix_af_true, q_matrix_af_pred)
    rmse_vs_df, mae_vs_df, r2_vs_df = calculate_error_metrics(q_vs_df_true, q_vs_df_pred)

    # Print results with timestamp
    current_time = datetime(2025, 6, 25, 14, 37)  # 02:37 PM WAT
    print(f"Results as of {current_time.strftime('%Y-%m-%d %I:%M %p WAT')}:")
    print(f"VS AF Model - RMSE: {rmse_vs_af:.4f}, MAE: {mae_vs_af:.4f}, R²: {r2_vs_af:.4f}")
    print(f"Matrix AF Model - RMSE: {rmse_matrix_af:.4f}, MAE: {mae_matrix_af:.4f}, R²: {r2_matrix_af:.4f}")
    print(f"VS DF Model - RMSE: {rmse_vs_df:.4f}, MAE: {mae_vs_df:.4f}, R²: {r2_vs_df:.4f}")

    # Sensitivity analysis
    param_range = np.linspace(0.9, 1.1, 10)  # ±10% variation
    sensitivity = sensitivity_analysis(theta, q_vs_af_pred, param_range)

    # Visualize results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.scatter(theta, q_vs_af_pred, label='VS AF Predicted', alpha=0.5)
    plt.scatter(theta, q_matrix_af_pred, label='Matrix AF Predicted', alpha=0.5)
    plt.scatter(theta, q_vs_df_pred, label='VS DF Predicted', alpha=0.5)
    plt.xlabel('Theta')
    plt.ylabel('q')
    plt.legend()
    plt.title('Model Predictions vs Theta')

    plt.subplot(1, 2, 2)
    plt.plot(param_range, sensitivity, marker='o')
    plt.xlabel('Parameter Variation')
    plt.ylabel('Sensitivity')
    plt.title('Sensitivity Analysis')

    plt.tight_layout()
    plt.show()

    # Open canvas for further visualization or code execution
    print("A canvas panel is available for visualizing charts or executing additional code.")
