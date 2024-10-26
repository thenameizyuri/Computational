import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Define constants
hbar = 1.0545718e-34  # Planck's constant (J·s)
m = 9.10938356e-31    # Mass of electron (kg)
L = 1e-9              # Length of the well (1 nm)
N = 1000              # Number of discrete points

# Discretize space
x = np.linspace(0, L, N)
dx = x[1] - x[0]

# Define the potential (0 inside the well, large outside)
V0 = 1e5  # Potential outside the well (J)
V = np.zeros(N)
V[0], V[-1] = V0, V0  # Infinite potential walls

#constructiing the Hamiltonian matrix using finite difference method
H = np.zeros((N, N))
for i in range(1, N-1):
    H[i, i] = -2 / dx**2
    H[i, i+1] = H[i, i-1] = 1 / dx**2
H = -(hbar**2 / (2 * m)) * H + np.diag(V)

# Solving the eigenvalue problem
energies, wavefunctions = eigh(H)

#plotting the first three eigenstates
plt.figure(figsize=(10, 6))
for n in range(3):
    plt.plot(x, wavefunctions[:, n]**2, label=f'n={n+1}, E={energies[n]:.2e} J')
plt.title('Wavefunctions for a Particle in a 1D Potential Well')
plt.xlabel('Position (m)')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
