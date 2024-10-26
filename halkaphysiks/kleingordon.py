import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 1.0  # Speed of light
m = 1.0  # Mass of the particle
L = 10.0  # Length of the spatial domain
T = 10.0  # Total time
dx = 0.1  # Spatial step
dt = 0.01  # Time step

# Discretizing the space and time
x = np.arange(0, L, dx)
t = np.arange(0, T, dt)
Nx = len(x)
Nt = len(t)

# Initial conditions
phi = np.zeros((Nt, Nx))
phi[0, :] = np.sin(np.pi * x / L)  # Initial wave profile
phi[1, :] = phi[0, :]  # Assume stationary initial condition (constant hatauna lai)

# finite difference method
for n in range(1, Nt - 1):
    for i in range(1, Nx - 1):
        phi[n+1, i] = (2 * phi[n, i] - phi[n-1, i]
                       + (c * dt / dx)**2 * (phi[n, i+1] - 2 * phi[n, i] + phi[n, i-1])
                       - (m * c * dt)**2 * phi[n, i])

# Plotting the results for different time slices
plt.figure(figsize=(12, 6))
for i in range(0, Nt, Nt // 5):
    plt.plot(x, phi[i, :], label=f't = {i*dt:.2f}')
plt.title('Numerical Solution of the Klein-Gordon Equation')
plt.xlabel('Position (x)')
plt.ylabel('Field (φ)')
plt.legend()
plt.show()
