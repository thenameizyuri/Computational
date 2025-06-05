import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# LaTeX formatting
rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
rc('text', usetex=True)

# Parameters
phi = 0.4
D_eff = 1e-7
rho_s = 2.5
K_F = 0.1
n = 0.7
R = 0.1            # Particle radius (cm)
c_bulk = 1.0       # Bulk concentration (mol/L)
c0 = 0.0           # Initial concentration (mol/L)

# Discretization
Nr = 50            # Spatial grid points
dr = R / Nr        # Spatial step
dt = 100           # Temporal step (s)
Nt = 1000          # Number of time steps

# Initialize grid
r = np.linspace(0, R, Nr+1)  # <-- Define 'r' here
c = np.zeros((Nr+1, Nt+1))
c[:, 0] = c0       # Initial condition

# Finite difference loop (critical missing part!)
for k in range(Nt):
    for i in range(1, Nr):
        denom = phi + (1 - phi) * rho_s * K_F * c[i, k]**(n-1)
        alpha = (phi * D_eff) / denom
        
        d2cdr2 = (c[i+1, k] - 2*c[i, k] + c[i-1, k]) / dr**2
        dcdr = (c[i+1, k] - c[i-1, k]) / (2 * dr)
        
        c[i, k+1] = c[i, k] + dt * alpha * (d2cdr2 + (2/r[i]) * dcdr)
    
    # Boundary conditions
    c[0, k+1] = c[1, k+1]      # Symmetry at r=0
    c[-1, k+1] = c_bulk        # Bulk concentration at r=R

# Plotting adjustments
plt.figure(figsize=(6, 4.5), dpi=300)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Fixed color codes
time_points = [0, 100, 500, 1000]
labels = [r'$t = 0$ s', r'$t = 10^4$ s', r'$t = 5 \times 10^4$ s', r'$t = 10^5$ s']  # Fixed quotes

for idx, k in enumerate(time_points):
    plt.plot(r, c[:, k],
             color=colors[idx],  # Fixed syntax: color=colors[idx]
             linewidth=1.5,
             linestyle='-',
             marker='o' if idx == 0 else '',
             markersize=4,
             label=labels[idx])

# Axis labels and grid
plt.xlabel(r'\textbf{Radial Position (cm)}', fontsize=12)  # Fixed LaTeX syntax
plt.ylabel(r'\textbf{Concentration (mol L$^{-1}$)}', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()