# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:06:03 2025

@author: LENOVO
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 25 14:58:08 2025

@author: LENOVO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gamma

# === Load incidence data ===
data = pd.read_csv('Python codes/Delta_provience.csv')  # Load data from CSV file
I = data.iloc[:, 9].values  # Assumes incidence is in the second column (index 1)
I = I[:200]  # Take first 300 days

plt.figure()
plt.plot(I)
plt.title('Incidence Data')
plt.xlabel('Time (days)')
plt.ylabel('Cases')
plt.show()

# === Serial interval (mean and SD) ===
mean_si = 5.2675
sd_si = 1.6025

# === Gamma distribution parameters ===
shape = (mean_si / sd_si)**2
scale = (sd_si**2) / mean_si

# === Simulation settings ===
T = len(I)
tau = 15  # Smoothing window
a_prior = 1
b_prior = 5

a_post = np.full(T, np.nan)
b_post = np.full(T, np.nan)

# === Compute lambda_t and posterior parameters ===
for t in range(tau, T):
    lambda_t = 0
    for s in range(t):
        delta_t = t - s
        w_s = gamma.pdf(delta_t, a=shape, scale=scale)  # Gamma weight for time difference
        lambda_t += I[s] * w_s
    
    if lambda_t > 0:
        a_post[t] = a_prior + I[t]
        b_post[t] = 1 / (1/b_prior + lambda_t)

# === Function to estimate R_t with credible intervals ===
def gamma_dis(a_post, b_post, nsamples=1000):
    Rt_mean = np.full_like(a_post, np.nan)
    Rt_median = np.full_like(a_post, np.nan)
    Rt_upper = np.full_like(a_post, np.nan)
    Rt_lower = np.full_like(a_post, np.nan)
    
    for t in range(len(a_post)):
        if not np.isnan(a_post[t]):
            samples = gamma.rvs(a=a_post[t], scale=b_post[t], size=nsamples)
            Rt_mean[t] = np.mean(samples)
            Rt_median[t] = np.median(samples)
            Rt_upper[t] = np.percentile(samples, 97.5)
            Rt_lower[t] = np.percentile(samples, 2.5)
    
    return Rt_mean, Rt_median, Rt_upper, Rt_lower

# === Estimate R_t ===
Rt_mean, Rt_median, Rt_upper, Rt_lower = gamma_dis(a_post, b_post)

# === Plot R_t and 95% CrI ===
valid_idx = ~np.isnan(Rt_lower) & ~np.isnan(Rt_upper) & ~np.isnan(Rt_mean)
t_vals = np.where(valid_idx)[0]
mean_vals = Rt_mean[valid_idx]
lower_vals = Rt_lower[valid_idx]
upper_vals = Rt_upper[valid_idx]

plt.figure()
plt.fill_between(t_vals, lower_vals, upper_vals, color='lightblue', alpha=1, label='95% CrI')
plt.plot(t_vals, mean_vals, 'k-', linewidth=2, label='Mean R_t')
plt.axhline(1, linestyle='--', color='red', linewidth=2.0, label='R_t = 1 threshold')
plt.xlabel('Time (days)')
plt.ylabel('R_t')
plt.title('Time-dependent Reproduction Number (R_t) with 95% Credible Interval')
plt.legend()
plt.grid(True)
plt.show()