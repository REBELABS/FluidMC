# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 08:13:43 2025

@author: user
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Simulated right-skewed volume flux data
q_data = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
log_q_data = np.log(q_data)

# Model A: LogNormal
with pm.Model() as model_lognormal:
    mu_ln = pm.Normal("mu", mu=0, sigma=1)
    sigma_ln = pm.HalfNormal("sigma", sigma=1)
    q_obs_ln = pm.LogNormal("q_obs", mu=mu_ln, sigma=sigma_ln, observed=q_data)
    trace_ln = pm.sample(1000, tune=1000, chains=2, progressbar=False, return_inferencedata=True)

# Model B: Normal on log(q)
with pm.Model() as model_normal_logq:
    mu_n = pm.Normal("mu", mu=0, sigma=1)
    sigma_n = pm.HalfNormal("sigma", sigma=1)
    q_obs_n = pm.Normal("q_obs", mu=mu_n, sigma=sigma_n, observed=log_q_data)
    trace_n = pm.sample(1000, tune=1000, chains=2, progressbar=False, return_inferencedata=True)

# Plot posterior for both models
fig1 = az.plot_posterior(trace_ln, var_names=["mu", "sigma"], hdi_prob=0.95)
plt.suptitle("Model A: LogNormal Likelihood", fontsize=14)

fig2 = az.plot_posterior(trace_n, var_names=["mu", "sigma"], hdi_prob=0.95)
plt.suptitle("Model B: Normal on log(q)", fontsize=14)

plt.show()