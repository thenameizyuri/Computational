"""
# SVIR Model#

dSdt = -βd * S * I - βi * S * V - d * S + η * R           
dIdt = βd * S * I + βi * S * V - γ * I - d * I           
dRdt = γ * I - d * R - η * R                              
dVdt = p * I - Omega_T * V

"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from math import exp, log #natural logarithm

# Define the model
def avian_model(y, t, params):
    S, I, R, V = y  # State variables
    βd, βi, d, η, γ, p, T, aT, bT = params #parameters

    # Temperature-dependent viral decay rate in water
    Omega_T = log(10) * exp(-aT * T - bT)  # Approximates log(10) * exp(-aT*T - bT)

    # Differential equations
    dSdt = -βd * S * I - βi * S * V - d * S + η * R           # Susceptible
    dIdt = βd * S * I + βi * S * V - γ * I - d * I            # Infected
    dRdt = γ * I - d * R - η * R                              # Recovered
    dVdt = p * I - Omega_T * V                                # Virus concentration

    return [dSdt, dIdt, dRdt, dVdt]
 
# ----------------------
# Parameters
# ----------------------
βd = 2.14e-9    # Direct transmission rate (bird-to-bird)
βi = 3.55e-9   # Indirect transmission rate (via water)
d = 0.1/365    # Natural death rate
η = 0.38       # Immunity loss rate
γ = 0.14       # Recovery rate
p = 1000        # Viral shedding rate into water
aT = -0.12     # Temperature decay parameter
bT = 5.10       # Environmental decay parameter

# Temperature range to analyze (°C)
temperatures = [4, 10, 20, 30]  # Cold to warm conditions

# ----------------------
# Initial Conditions
# ----------------------
S0 = 10000      # Initial susceptible birds
I0 = 10         # Initial infected birds
R0 = 0          # Initial recovered birds
V0 = 10**(4.7)  # Initial virus concentration in water

# Time vector (in days, representing one season)
t = np.linspace(0, 365, 1000)

# ----------------------
# Simulation and Plotting
# ----------------------
plt.figure(figsize=(14, 10))
colors = ['b', 'g', 'r', 'm']  # Different colors for different temperatures

# Plot for each temperature
for i, T in enumerate(temperatures):
    params = (βd, βi, d, η, γ, p, T, aT, bT)
    solution = odeint(avian_model, [S0, I0, R0, V0], t, args=(params,))
    S, I, R, V = solution.T
    
    # Calculate Ω(T) for this temperature
    Omega_T = 2.3026 * np.exp(-aT * T - bT)
    
    # Correct R0 calculation as sum of direct and indirect components
    R0d = (βd * S0) / (γ + d)  # Direct transmission component
    R0i = (βi * p * S0) / ((γ + d) * Omega_T)  # Indirect transmission component
    R0 = R0d + R0i  # Total basic reproduction number
    
    # Plot all variables
    plt.subplot(2, 2, 1)
    plt.plot(t, S, colors[i], label=f'{T}°C (R0={R0:.2f})')
    
    plt.subplot(2, 2, 2)
    plt.plot(t, I, colors[i], label=f'{T}°C')
    
    plt.subplot(2, 2, 3)
    plt.plot(t, R, colors[i], label=f'{T}°C')
    
    plt.subplot(2, 2, 4)
    plt.plot(t, V, colors[i], label=f'{T}°C')

# Formatting plots
plt.subplot(2, 2, 1)
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('Susceptible Birds (S)')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('Infected Birds (I)')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('Recovered Birds (R)')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.xlabel('Time (days)')
plt.ylabel('Virus Load')
plt.title('Environmental Virus (V)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# ----------------------
# Impact of temperature variation on basic reproduction number R0
# Temperature Sensitivity Analysis
# ----------------------
# Create a range of temperatures for sensitivity analysis
# Temperature range (0°C to 40°C)
T_range = np.linspace(0, 40, 100)

# Calculate R0 components
Omega_T = log(10) * np.exp(-aT * T_range - bT) # Virus decay rate in environment

R0_total = (βd * S0)/(γ + d) + (βi * p * S0)/((γ + d) * Omega_T) 

# Create the plot
plt.figure(figsize=(10, 5))
plt.plot(T_range, R0_total, 'r-', linewidth=3, label='Total R0')

# Add epidemic threshold line
plt.axhline(y=1, color='k', linestyle='--', label='Epidemic threshold (R0=1)')

# Highlight key temperature points
for T in [4, 10, 20, 30]:
    idx = np.argmin(np.abs(T_range - T))
    plt.plot(T, R0_total[idx], 'ro', markersize=8)
    plt.text(T, R0_total[idx]+0.2, f'{R0_total[idx]:.2f}', 
             ha='center', va='bottom', fontsize=10)

# Formatting
plt.title('Basic Reproduction Number (R0) vs Temperature', fontsize=14)
plt.xlabel('Temperature (°C)', fontsize=12)
plt.ylabel('R0', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, loc='upper right')
plt.xlim(0, 40)
plt.ylim(0, max(R0_total)*1.1)

plt.tight_layout()
plt.show()