
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 04:30:45 2024

@author: LENOVO
"""


import numpy as np
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def spir_model(y, t, a, b, c, d):
    S, P, I, R = y
    dSdt = -a * S * I - b * S
    dPdt = b * S - c * P * I
    dIdt = a * S * I + c * P * I - d * I
    dRdt = d * I
    return [dSdt, dPdt, dIdt, dRdt]

def fit_spir(t, S0, P0, I0, R0, a, b, c, d):
    y0 = [S0, P0, I0, R0]
    solution = odeint(spir_model, y0, t, args=(a, b, c, d))
    return solution[:, 2]

def objective(params, t, I_data, S0, P0, I0, R0,d):
    a, b, c = params
    I_model = fit_spir(t, S0, P0, I0, R0, a, b, c, d)
    return np.sum((I_model - I_data) ** 2)

# Read data
Idata = pd.read_csv("Python codes/SPIR2/mydataH1N1.csv")

# Extract the infected data
I_data = Idata.iloc[:, 1].values


# Initial values
S0, P0, I0, R0 = 18223, 0, 11, 0

# Fixed parameter
d = 1/6
# Initial guess for parameters a, b, c, d
initial_guess = [5.45e-6, 0.2, 3.74e-7]

# Time points
t = np.linspace(0, 40, 40)

# Fit the model to data
result = minimize(objective, initial_guess, args=(t, I_data, S0, P0, I0, R0, d), bounds=[(0, 1), (0, 1), (0, 1)])
fitted_a, fitted_b, fitted_c = result.x

# Solve the SPIR model with the fitted parameters
solution = odeint(spir_model, [S0, P0, I0, R0], t, args=(fitted_a, fitted_b, fitted_c, d))

# Plot the results
plt.figure(figsize=(6, 4))
plt.plot(t, I_data, 'r.',label='Actual Infected Data')
plt.plot(t, solution[:, 2], 'g--',label='Fitted Infected')
plt.xlabel('Time in days')
plt.ylabel('Population')
plt.title('SPIR Model Simulation and Fitting')
plt.legend()
plt.show()

print(f'Fitted a: {fitted_a}')
print(f'Fitted b: {fitted_b}')
print(f'Fitted c: {fitted_c}')

