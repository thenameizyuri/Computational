import numpy as np
import matplotlib.pyplot as plt

# Constants
L = 1.0  # Width of the well in meters
n_states = 3  # Number of energy levels to plot
x = np.linspace(0, L, 1000)  # Position values

# Function to calculate the wave function
def wave_function(n, x, L):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

# Plot the wave functions for the first few states
plt.figure(figsize=(10, 6))
for n in range(1, n_states + 1):
    psi_n = wave_function(n, x, L)
    plt.plot(x, psi_n, label=f'n = {n}')

plt.title('Wave Functions for a Particle in a 1D Infinite Potential Well')
plt.xlabel('Position (x)')
plt.ylabel('Wave Function (ψ)')
plt.legend()
plt.grid()
plt.show()
