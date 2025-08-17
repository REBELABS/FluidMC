# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 02:26:50 2025

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import halfnorm, truncnorm
from scipy.stats import beta
from scipy.stats import norm

#Distributions consider for the baseflux
# Range of x values for plotting
x = np.linspace(0, 0.1, 1000)

# HalfNormal: sigma = 1
halfnorm_pdf = halfnorm(scale=1).pdf(x)

# TruncatedNormal: mu=0.00001, sigma=0.0001, lower=0
#A tuple unbundled into a and b respectively for the lower and upper bound
a, b = (0 - 0.00001) / 0.0001, np.inf  # Standardized bounds for truncnorm
truncnorm_pdf = truncnorm(a=a, b=b, loc=0.00001, scale=0.0001).pdf(x)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(x, halfnorm_pdf, label="HalfNormal(sigma=1)", lw=2)
plt.plot(x, truncnorm_pdf, label="TruncatedNormal(mu=0.01, sigma=0.01, lower=0)", lw=2)
plt.title("Comparison of Priors for `baseflux`")
plt.xlabel("baseflux")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()


#Distributions consider for the theta
# Define normalized theta range [0, 1]
theta_norm = np.linspace(0, 1, 1000)

# Define PDFs
uniform_pdf = np.ones_like(theta_norm)  # Uniform(0, 1)

beta_weak_pdf = beta(1.1, 1.1).pdf(theta_norm)  # Near-uniform
jeffrey_pdf = beta(0.5, 0.5).pdf(theta_norm)    # Jeffreys prior (edge-weighted)

# Plot
plt.figure(figsize=(9, 5))
plt.plot(theta_norm, uniform_pdf, label="Uniform(0, 1)", lw=2)
plt.plot(theta_norm, beta_weak_pdf, label="Beta(1.1, 1.1)", lw=2, color="orange")
plt.plot(theta_norm, jeffrey_pdf, label="Jeffreys: Beta(0.5, 0.5)", lw=2, color="green")
plt.title("Priors for Normalized θ ∈ [0, 1]")
plt.xlabel("Normalized θ")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Define range of values for weights
x = np.linspace(-4, 4, 1000)

# Normal(0, 1) for both w_theta and w_phi
pdf = norm(loc=0, scale=1).pdf(x)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, pdf, lw=2, label="Normal(μ=0, σ=1)")
plt.title("Prior for w_theta and w_phi")
plt.xlabel("Weight value")
plt.ylabel("Probability Density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



# Define range for sigma_q values
x = np.linspace(0, 5, 1000)

# HalfNormal with sigma=1
pdf = halfnorm(scale=0.2).pdf(x)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, pdf, lw=2, label="HalfNormal(σ=1)")
plt.title("Prior for σ_q (Standard Deviation of q)")
plt.xlabel("σ_q")
plt.ylabel("Probability Density")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()