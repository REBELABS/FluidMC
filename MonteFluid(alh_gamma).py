# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 10:31:37 2025

@author: REBELABS
"""
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import arviz as az
import pandas as pd

# Set random seed
np.random.seed(42)

# Setup variables
alh_pr = np.array([0.7508, 0.6868, 0.5952])
theta = np.array([0, 20, 50, 70, 90])
q_AB = np.array([0.0047, 0.0071, 0.0518, 0.0506, 0.0071])
q_AC = np.array([0.0071, 0.0071, 0.0542, 0.0495, 0.0141])
q_AD = np.array([0.0141, 0.0153, 0.0130, 0.0683, 0.0612])

# Combine all ALH experiments
theta_all = np.tile(theta, 3)
phi_r_all = np.concatenate([np.full(len(theta), p) for p in alh_pr])
q_all = np.concatenate([q_AB, q_AC, q_AD])

# Normalize inputs
theta_norm = (theta_all - theta_all.min()) / (theta_all.max() - theta_all.min())
phi_r_min = alh_pr.min()
phi_r_max = alh_pr.max()
phi_r_norm = (phi_r_all - phi_r_min) / (phi_r_max - phi_r_min)

# Build model
with pm.Model() as cond_density_model_alh:
    theta_data = pm.Data("theta", theta_norm)
    phi_r_data = pm.Data("phi_r", phi_r_norm)
    q_obs_data = pm.Data("q_obs", q_all)

    baseflux = pm.HalfNormal("baseflux", sigma=0.01)
    w_theta = pm.Normal("w_theta", mu=0, sigma=1)
    w_phi = pm.Normal("w_phi", mu=0, sigma=1)
    w_inter = pm.Normal("w_inter", mu=0, sigma=1)

    mu_q = pm.Deterministic('mu_q',baseflux + w_theta * theta_data + w_phi * phi_r_data + w_inter * theta_data * phi_r_data)

    sigma_q = pm.Exponential("sigma_q", lam=10.0)

    alpha = mu_q**2 / sigma_q**2
    beta = mu_q / sigma_q**2
    q_likelihood = pm.Gamma("q_likelihood", alpha=alpha, beta=beta, observed=q_obs_data)

    trace_alh = pm.sample(
        5000, tune=2000, target_accept=0.97, return_inferencedata=True, random_seed=42
    )
    post_pred_alh = pm.sample_posterior_predictive(
        trace_alh, var_names=["q_likelihood"], return_inferencedata=True, random_seed=42
    )

# Plot posterior predictive
ppc = post_pred_alh.posterior_predictive["q_likelihood"].values.flatten()
plt.figure(figsize=(14, 10))
plt.hist(ppc, bins='auto', density=True, alpha=0.35, label="Posterior predictive q")
plt.hist(q_all, bins='auto', density=True, alpha=0.6, edgecolor="k", label="Observed lab q")
plt.xlabel("q")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("Predictive_Post_plot_alh5", dpi=300, bbox_inches='tight')
plt.show()

# Prior predictive
with cond_density_model_alh:
    pior_pred_alh = pm.sample_prior_predictive(draws=3000,
                                               var_names=['q_likelihood'],
                                               random_seed=42, return_inferencedata=True)

prior_samples = pior_pred_alh.prior_predictive["q_likelihood"].values.flatten()
plt.figure(figsize=(14, 10))
plt.hist(prior_samples, bins=100, density=True, alpha=0.4, label="Prior Predictive q")
plt.hist(q_all, bins=30, density=True, alpha=0.6, edgecolor="k", label="Observed lab q")
plt.xlabel("q", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.xlim(0, 1.2)
plt.legend()
plt.tight_layout()
plt.savefig("Prior_Predictive_Histogram_ALH5.png", dpi=300, bbox_inches='tight')
plt.show()

# Trace plots
chain_colors = ['red','blue','orange','green']
alh_pplot = ['w_theta','w_phi','baseflux','sigma_q']
fig_trace_alh = az.plot_trace(trace_alh, var_names=alh_pplot, figsize=(14,10), chain_prop={'color':chain_colors}, legend=True)
for i, ax in enumerate(fig_trace_alh.flatten()):
    if i % 2 == 0:
        ax.set_ylabel("Density")
        ax.set_xlabel("Sample value")
    else:
        ax.set_ylabel("Raw sampled value")
        ax.set_xlabel("Sampling step")
plt.tight_layout()
plt.savefig("Trace_plot_alh3", dpi=300, bbox_inches='tight')
plt.show()

# Summary
az.summary(trace_alh).to_csv('alh_Diagnostic_Summary.csv')

# Posterior plots
fig_post_alh = az.plot_posterior(trace_alh, var_names=alh_pplot, figsize=(14,10), hdi_prob=0.95, kind='kde')
for ax in fig_post_alh.flatten():
    ax.set_xlabel("Parameter value")
    ax.set_ylabel("Posterior density")
plt.tight_layout()
plt.savefig("Posterior_plot_alh5", dpi=300, bbox_inches='tight')
plt.show()

# ArviZ PPC
alh_pairss = {'q_likelihood': 'q_likelihood'}
fig_post_pred_alh = az.plot_ppc(post_pred_alh, data_pairs=alh_pairss, var_names=['q_likelihood'], figsize=(14,10), group='posterior', num_pp_samples=200)
fig_post_pred_alh.set_xlim(0,0.1)
fig_post_pred_alh.set_ylabel("Density", fontsize=12)
fig_post_pred_alh.set_xlabel("Sample value", fontsize=12)
plt.tight_layout()
plt.savefig("Predictive_Post_plot_alh_ppc5", dpi=300, bbox_inches='tight')
plt.show()

# Predictive performance
ppc_r = post_pred_alh.posterior_predictive["q_likelihood"].stack(s=("chain","draw")).values
ppc_r_mean = ppc_r.mean(axis=1)
ppc_r_lo = np.quantile(ppc_r, 0.025, axis=1)
ppc_r_hi = np.quantile(ppc_r, 0.975, axis=1)
coverage = ((q_all >= ppc_r_lo) & (q_all <= ppc_r_hi)).mean()
rmse = np.sqrt(np.mean((ppc_r_mean - q_all)**2))
print(f"✅ 95% predictive coverage: {coverage:.1%}")
print(f"✅ RMSE: {rmse:.5f}")
with open("confidenc_alh.txt","w") as f:
#Save the output to a log .txt file
    f.write("ALH\n")
    f.write(f"95% predictive coverage: {coverage:.1%}\n")
    f.write(f"✅ RMSE: {rmse:.5f}\n")

##Comaprison
#Declare variables
n_samples = 1000  # how many pairs you want
theta_rand = np.random.uniform(0, 90, size=n_samples)
phi_r_rand = np.random.uniform(0.5952, 0.7508, size=n_samples)

# equations from literature review
def v_alh(theta):
    #VS model for Ascending Flow (AF)
    return (-1.50e-6) * theta**3 + 1.76e-4 * theta**2 - 3.05e-3 * theta + 2.75e-2

def m_alh(theta):
    #Matrix model for Ascending Flow (AF)
    return (6.04e-3) * theta - 1.04e-16

def s_alh(theta, phi_r, c):
    c =1
    x = theta * phi_r * c
    y = theta * c
    exp_q = 0.9015 - 0.00190 * theta + 0.2033 * phi_r - 0.0049 * x + 0.0055 * y
    return np.exp(exp_q)

#Answer holders
v_space =[]
m_space = []
exp_space = []

#Model loop
for thetaa, phii in zip(theta_rand, phi_r_rand):
    vq = v_alh(thetaa)
    mq = m_alh(thetaa)
    eq = s_alh(thetaa,phii,c=1)
    
    #Append the ansers to the blank list
    v_space.append(vq)
    m_space.append(mq)
    exp_space.append(eq)

#Table for the estimated q
df_models = pd.DataFrame({'Theta':theta_rand,'Phi_r':phi_r_rand,
                          'v_alh (q)':v_space,'m_alh (q)':m_space,
                          's_alh (q)':exp_space})
    
print(df_models)  


#Normalizing  the input parameters since same was done for training bayesian model
#Recall use same max and min as used for the bay's training 
theta_norm_rand = (theta_rand-theta_all.min())/(theta_all.max()-theta_all.min())
phi_r_norm_rand = (phi_r_rand-alh_pr.min())/(alh_pr.max()-alh_pr.min())

#Input into the bayesian model as a conditional to get the qs
#Draw posteriro predictive samples for the new inputs, based on earlier training 
#result trace_alh, only get values for q_obs
with cond_density_model_alh:
    #load in the values
    pm.set_data({'theta': theta_norm_rand,'phi_r': phi_r_norm_rand})
    mu_pred = pm.sample_posterior_predictive(trace_alh,
                                            var_names=['mu_q'],
                                            return_inferencedata=True,
                                            random_seed=42,predictions=True)
#Get out the generated values from the Posterior Predictive Values
mu_pred_values = mu_pred.predictions["mu_q"].stack(samples=("chain", "draw")).values
#print(mu_pred_values)
mu_mean = mu_pred_values.mean(axis=1)                     # Posterior mean for each input
mu_lower = np.quantile(mu_pred_values, 0.025, axis=1)     # Lower bound (2.5% quantile)
mu_upper = np.quantile(mu_pred_values, 0.975, axis=1)     # Upper bound (97.5% quantile)

#Append to the created DF
df_models['Bay (q)']=mu_mean
df_models['Bay Lower']=mu_lower
df_models['Bay Upper']=mu_upper

#Export the CSV
df_models.to_csv('Mean Comparison.csv', index=False)



#sigma_q_raw = pm.Exponential("sigma_q_raw", lam=10.0)
#sigma_q = pm.math.clip(sigma_q_raw, 1e-3, np.inf)
