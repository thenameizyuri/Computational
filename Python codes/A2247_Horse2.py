# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 07:12:31 2025

@author: LENOVO
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 01:43:30 2025

@author: LENOVO
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

# Load data
data = pd.read_csv("Python codes/A2247_Horse2.csv")  # Assumes headers exist
t_data = data.iloc[:, 1].values  # Original time values
V_data = data.iloc[:, 2].values      # Virus data (log10 scale)

# Adjust time to start from 0 while preserving experimental time points
  # Time since first measurement
t_span = (0, 40)                     # Simulation time span (0-50 days)

# Convert viral load from log10 to linear scale for fitting
#V_data_linear = 10**V_data

# Fixed parameters (from your input)
delta = 1/21
gamma = 10.88
rho = 1/21
sigma = 0
alpha = 0
lam = 2019

# Initial conditions
M0 = 42390
I0 = 0
V0 = 467
y0 = [M0, I0, V0]

# System of ODEs
def system(t, y, beta, b):
    M, I, V = y
    dMdt = lam - rho*M - beta*M*V
    dIdt = beta*M*V - delta*I - sigma*I
    dVdt = b*I - gamma*V - alpha*V
    return [dMdt, dIdt, dVdt]

# Objective function for least squares fitting
def objective(params, t_data, V_data_log):
    beta, b = params
    
    # Solve ODE with current parameters across full 50-day span
    sol = solve_ivp(
        lambda t, y: system(t, y, beta, b),
        t_span,
        y0,
        t_eval=t_data,  # Only compare at experimental time points
        method='LSODA'
    )
    
    # Calculate predicted virus (log10 scale)
    V_pred = np.log10(sol.y[2] + 1e-10)  # Avoid log(0)
    
    # Return sum of squared errors at experimental time points
    return np.sum((V_pred - V_data_log)**2)

# Initial guesses for parameters to estimate
initial_guess = [1.98e-7, 889]  # [beta, b]

# Parameter bounds (optional but recommended)
bounds = [
    (1e-9, 870),  # beta
    (1, 900)    # b
]

# Perform optimization
result = minimize(
    objective,
    initial_guess,
    args=(t_data, V_data),
    bounds=bounds,
    method='L-BFGS-B'
)

# Extract fitted parameters
beta_fit, b_fit = result.x


# Full simulation with fitted parameters (0-50 days)
t_eval = np.linspace(0, 30, 500)  # High resolution for smooth plot
sol_fit = solve_ivp(
    lambda t, y: system(t, y, beta_fit, b_fit),
    t_span,
    y0,
    t_eval=t_eval,
    method='LSODA'
)

# Plot results
plt.figure(figsize=(10, 6))
# Experimental points (with original time offsets)
plt.scatter(t_data, V_data, color='red', label='Experimental Data', linewidth=7)
# Model prediction (full 50-day span)
plt.plot(sol_fit.t, np.log10(sol_fit.y[2]), 'b-', label='Fitted Model', linewidth= 4)
plt.xlabel('Time (days since first measurement)')
plt.ylabel(' (log10v RNA\ml plasma)', fontsize = 20)
plt.title(' Horse A2247', fontsize = 20)
plt.legend(fontsize=16)
plt.grid(True)
#plt.xlim(0, 50)  # Explicitly set x-axis limits
plt.show()
print(f"Fitted parameters:")
print(f"beta = {beta_fit:.3e} (infection rate)")
print(f"b = {b_fit:.1f} (virus production rate)")