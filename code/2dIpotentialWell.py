import numpy as np
import matplotlib.pyplot as plt

# Well dimensions
Lx = 1.0  # Width in the x-direction (m)
Ly = 1.0  # Width in the y-direction (m)
nx_states = 2  # Number of states in the x-direction
ny_states = 2  # Number of states in the y-direction

# Position values
x = np.linspace(0, Lx, 100)
y = np.linspace(0, Ly, 100)
X, Y = np.meshgrid(x, y)

# Function to calculate the wave function
def wave_function_2d(nx, ny, X, Y, Lx, Ly):
    psi_x = np.sqrt(2 / Lx) * np.sin(nx * np.pi * X / Lx)
    psi_y = np.sqrt(2 / Ly) * np.sin(ny * np.pi * Y / Ly)
    return psi_x * psi_y

#plot the probability densities of the first few states
fig, axs = plt.subplots(nx_states, ny_states, figsize=(12, 12))
fig.suptitle('_', fontsize=16)

for i in range(nx_states):
    for j in range(ny_states):
        nx = i + 1
        ny = j + 1
        psi_2d = wave_function_2d(nx, ny, X, Y, Lx, Ly)
        probability_density = np.abs(psi_2d)**2
        ax = axs[i, j]
        c = ax.contourf(X, Y, probability_density, levels=50, cmap='viridis')
        ax.set_title(f'nx = {nx}, ny = {ny}')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        fig.colorbar(c, ax=ax)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
