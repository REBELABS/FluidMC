# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:31:37 2025

@author: REBELABS
"""
import numpy as np
from scipy import stats
import pymc as pm
import matplotlib.pyplot as plt
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

##Setup the variables
#Porosity Ratio (Phil_r)
alh_pr = np.array([0.7508,0.6868,0.5952])
dlh_pr = np.array([0.0133,1.456,1.68])

#Outlet Angle (Theta)
theta = np.array([0,20,50,70,90])

#ALH Volume Flux
q_AB = np.array([0.0047,0.0071,0.0518,0.0506,0.0071])
q_AC = np.array([0.0071,0.0071,0.0542,0.0495,0.0141])
q_AD = np.array([0.0141,0.0153,0.0130,0.0683,0.0612])

#DLH Volume Flux
q_BA = np.array([0.1543,0.1095,0.0448,0.0247,0])
q_CA = np.array([0.1543,0.1272,0.0730,0.0306,0])
q_DA = np.array([0.1760,0.1398,0.0836,0.0836,0])

#Creating a triplet for eaiser numerations
#for samle AB
data_AB = []
for i in range(len(theta)):
    entry = {'theta':theta[i],'phi_r':alh_pr[0],'q':q_AB[i]}
    data_AB.append(entry)

#Creating a triplet for samle AC
data_AC = []
for i in range(len(theta)):
    entry = {'theta':theta[i],'phi_r':alh_pr[1],'q':q_AC[i]}
    data_AC.append(entry)
    
#Creating a triplet for samle AD
data_AD = []
for i in range(len(theta)):
    entry = {'theta':theta[i],'phi_r':alh_pr[2],'q':q_AD[i]}
    data_AD.append(entry)

##Get out the values and make numpy array again for fast computation and pymc demand
#Get out the values for data_AB
theta_AB = np.array([item['theta'] for item in data_AB])
phi_r_AB = np.array([item['phi_r'] for item in data_AB])
vq_AB = np.array([item['q'] for item in data_AB])

#Get out the values for data_AC
theta_AC = np.array([item['theta'] for item in data_AC])
phi_r_AC = np.array([item['phi_r'] for item in data_AC])
vq_AC = np.array([item['q'] for item in data_AC])

#Get out the values for data_AD
theta_AD = np.array([item['theta'] for item in data_AD])
phi_r_AD = np.array([item['phi_r'] for item in data_AD])
vq_AD = np.array([item['q'] for item in data_AD])

##Normalize the values. Helps with weak piriors, improve geometric samler (NUT)
##& numerical stability for fast convergence. #Min to Max normalization range (0 to 1)

#Setting the global min and max phi_r to avoid division by zero
phi_r_min = alh_pr.min()
phi_r_max = alh_pr.max()

#Sample AB
theta_AB_norm = (theta_AB-theta_AB.min())/(theta_AB.max()-theta_AB.min())
phi_r_AB_norm = (phi_r_AB-phi_r_min)/(phi_r_max-phi_r_min)

#Sample AC
theta_AC_norm = (theta_AC-theta_AC.min())/(theta_AC.max()-theta_AC.min())
phi_r_AC_norm = (phi_r_AC-phi_r_min)/(phi_r_max-phi_r_min)

#Sample AD
theta_AD_norm = (theta_AD-theta_AD.min())/(theta_AD.max()-theta_AD.min())
phi_r_AD_norm = (phi_r_AD-phi_r_min)/(phi_r_max-phi_r_min)

#Concentate
theta_all_alh = np.concatenate([theta_AB_norm,theta_AC_norm,theta_AD_norm])
phi_r_all_alh = np.concatenate([phi_r_AB_norm,phi_r_AC_norm,phi_r_AD_norm])
q_all_alh = np.concatenate([vq_AB,vq_AC,vq_AD])

#Print to verify normalization
#print(f'{theta_all_alh}')
#print(f'{phi_r_all_alh}')
#print(f'{q_all_alh}')

with pm.Model() as cond_density_model_alh:
    #Declare the inputs
    theta_data = pm.MutableData("theta", theta_all_alh)
    phi_r_data = pm.MutableData("phi_r", phi_r_all_alh)
    q_data = pm.MutableData("q", q_all_alh)
    
    #Pirior of the baseline q, i.e q when theta and porosity ratio are nuetral or close to zero
    # TruncatedNormal: mu=0.0001, sigma=0.001, lower=0
    baseflux = pm.TruncatedNormal("baseflux", mu =0.0001, sigma = 0.001, lower =0)
    
    #Pirior for the outlet angle. This is applied on the regression weights not the actual values
    #it allows for the mixed relationship with q. Model learn from the data whether increasing theta or phi_r contributes + or - to q
    w_theta = pm.Normal("w_theta", mu=0, sigma=1)

    #Pirior for the porosity ratio.
    w_phi = pm.Normal("w_phi", mu=0, sigma=1)
    
    #Std around the predicted q or messiness of the q
    sigma_q = pm.Half





#PYMC
with pm.Model() as gp_model:
    


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
