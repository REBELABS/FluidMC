# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:31:37 2025

@author: REBELABS
"""
import numpy as np
from scipy import stats
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd
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

#investigate collinearity for the conditionals
theta_co = np.tile(theta_AB,3) #Since theta is repeated, this is used to concenta
phi_r_co = np.concatenate([phi_r_AB,phi_r_AC,phi_r_AD])
diif = pd.DataFrame({'theta_co': theta_co,'phi_r_co': phi_r_co})
diif.corr()

#Sample AB
theta_AB_norm = (theta_AB-theta_AB.min())/(theta_AB.max()-theta_AB.min())
phi_r_AB_norm = (phi_r_AB-phi_r_min)/(phi_r_max-phi_r_min)

#Sample AC
phi_r_AC_norm = (phi_r_AC-phi_r_min)/(phi_r_max-phi_r_min)

#Sample AD
phi_r_AD_norm = (phi_r_AD-phi_r_min)/(phi_r_max-phi_r_min)

#Concentate
theta_all_alh = np.tile(theta_AB_norm,3) #Since theta is repeated, this is used to concenta
phi_r_all_alh = np.concatenate([phi_r_AB_norm,phi_r_AC_norm,phi_r_AD_norm])
q_all_alh = np.concatenate([vq_AB,vq_AC,vq_AD])

#Print to verify normalization
#print(f'{theta_all_alh}')
#print(f'{phi_r_all_alh}')
#print(f'{q_all_alh}')

with pm.Model() as cond_density_model_alh:
    #Declare the inputs
    theta_data = pm.Data("theta", theta_all_alh)
    phi_r_data = pm.Data("phi_r", phi_r_all_alh)
    q_data = pm.Data("q", q_all_alh)
    
    #Pirior of the baseline q (mean not dist.), i.e q when theta and porosity ratio are nuetral or close to zero
    # TruncatedNormal: mu=0.00001, sigma=0.0001, lower=0
    baseflux = pm.TruncatedNormal("baseflux", mu =0.00001, sigma = 0.0001, lower =0)
    
    #Pirior for the outlet angle. This is applied on the regression weights not the actual values
    #it allows for the mixed relationship with q. Model learn from the data whether increasing theta or phi_r contributes + or - to q
    w_theta = pm.Normal("w_theta", mu=0, sigma=0.5)

    #Pirior for the porosity ratio.
    w_phi = pm.Normal("w_phi", mu=0, sigma=0.5)
    
    #Std around the observed q in real world at same inputs. USed in the likelihood function. 
    #Defines the likelihood’s variance, not a prior variance
    sigma_q = pm.HalfNormal("sigma_q",sigma=0.1)
    
    #Deterministic prediction of q based on physics and parameters before adding noise
    mu_q = pm.Deterministic("mu_q",baseflux + w_theta * theta_data + w_phi * phi_r_data)
    
    #Observed q based on parameters given
    #Lognormal used because large experiment has show q is right skewed. 
    #Widely when parameter is positive, used in fluid mechanics, transport modeling hydrology and output is multplicative depending other parameters 
    #sigma set to be flexible to allow the observed q match the experiment trend
    q_obs = pm.LogNormal("q_obs", mu = mu_q, sigma = sigma_q, observed = q_all_alh)
    

#Pirior Test
#Drawing samples from the Priors
with cond_density_model_alh:
    pior_pred_alh = pm.sample_prior_predictive(draws=3000,var_names=['q_obs'],
                                              random_seed=42,return_inferencedata=True)

#Predictive Prior plot
alh_pairs={'q_obs':'q_obs'}
fig_prior_alh = az.plot_ppc(pior_pred_alh,data_pairs=alh_pairs,var_names=['q_obs'],figsize=(14,10),group='prior')
fig_prior_alh.set_ylabel("Density", fontsize = 12)
fig_prior_alh.set_xlabel("Sample value", fontsize =12)
plt.tight_layout()
plt.savefig("Predictive_Prior_plot_alh", dpi = 300, bbox_inches = 'tight')
plt.show()

#print(np.min(pior_pred_alh.prior['mu_q'].values), np.max(pior_pred_alh.prior['mu_q'].values))

##Since Prior test checks out, now draw samples from posterior distribution
with cond_density_model_alh:
    trace_alh = pm.sample(3000, tune=1000, target_accept=0.95, return_inferencedata=True,
                          random_seed=42, progressbar=True)

#Trace plots (How the chains evolved over the parameters)
chain_colors = ['red','blue','orange','green']
alh_pplot = ['w_theta','w_phi','baseflux','sigma_q']
fig_trace_alh = az.plot_trace(trace_alh,alh_pplot,figsize=(14,10), chain_prop={'color':chain_colors},legend=True)
for i, ax in enumerate(fig_trace_alh.flatten()): #make the array easy to loop over type: np array (n,2)
    if i % 2 == 0:
        ax.set_ylabel("Density")
        ax.set_xlabel("Sample value")
    else:
        ax.set_ylabel("Raw sampled value")
        ax.set_xlabel("Sampling step")
plt.tight_layout()
plt.savefig("Trace_plot_alh", dpi = 300, bbox_inches = 'tight')
plt.show()

#Diagnostics Summary exported to csv
az.summary(trace_alh).to_csv('alh_Diagnostic_Summary.csv')

#Posterior plot (THe concluding shape of the parameters)
fig_post_alh = az.plot_posterior(trace_alh,alh_pplot,figsize=(14,10),
                             hdi_prob=0.95, kind='kde')
for i, ax_post in enumerate(fig_post_alh.flatten()): #make the array easy to loop over type: np array (n,2)
        ax_post.set_ylabel("Parameter value")
        ax_post.set_xlabel("Posterior density")
plt.tight_layout()
plt.savefig("Posterior_plot_alh", dpi = 300, bbox_inches = 'tight')
plt.show()

#Predictive Posterior plot
with cond_density_model_alh:
    post_pred_alh = pm.sample_posterior_predictive(trace_alh,var_names=['q_obs'],
                                                   random_seed=42,
                                                   return_inferencedata=True)

#Plot the posterior
alh_pairss={'q_obs':'q_obs'}
fig_post_pred_alh = az.plot_ppc(post_pred_alh,data_pairs=alh_pairss,
                                var_names=['q_obs'],figsize=(14,10),
                                group='posterior',num_pp_samples=200)
fig_post_pred_alh.set_xlim(0,0.1)
fig_post_pred_alh.set_ylabel("Density", fontsize = 12)
fig_post_pred_alh.set_xlabel("Sample value", fontsize =12)
plt.tight_layout()
plt.savefig("Predictive_Post_plot_alh", dpi = 300, bbox_inches = 'tight')
plt.show()

#Compare Lab q with Predictive Posterior q
ppc = post_pred_alh.posterior_predictive["q_obs"].values.reshape(-1)
plt.figure(figsize=(14,10))
plt.hist(ppc, bins='auto', density=True, alpha=0.35, label="Posterior predictive (q_obs)")
plt.hist(q_all_alh, bins='auto', density=True, alpha=0.6, edgecolor="k", label="Observed lab q")
plt.xlabel(r"q $(ms^{-1})$")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()

ppc_r = post_pred_alh.posterior_predictive["q_obs"].stack(s=("chain","draw")).values  # (N, S)

ppc_r_mean = ppc_r.mean(axis=1)
ppc_r_lo   = np.quantile(ppc_r, 0.025, axis=1)
ppc_r_hi   = np.quantile(ppc_r, 0.975, axis=1)

coverage = ((q_all_alh >= ppc_r_lo) & (q_all_alh <= ppc_r_hi)).mean()
rmse = np.sqrt(np.mean((ppc_r_mean - q_all_alh)**2))
print(f"95% predictive coverage: {coverage:.1%}")
print(f"RMSE: {rmse:.5f}")

print(az.__version__)

print(pm.__version__)


#PYMC
with pm.Model() as gp_model:
    



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
