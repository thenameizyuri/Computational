import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Parameters
L, N, dt = 5, 100, 0.001
dx = 2*L/N
x = np.linspace(-L, L, N+1)
t_max, M = 1, int((t_max)/dt)
g, V0, w, w0 = 1, 1, 1, 1

# Initial condition
sigma = 0.5
psi = np.exp(-x**2 / (2*sigma**2)) / (np.sqrt(np.pi) * sigma)**0.5
psi[0], psi[-1] = 0, 0  # Dirichlet BCs

# Potential
def V(x, t):
    return 0.5 * w0**2 * x**2 + V0 * np.cos(w * t)

# Tridiagonal matrix setup
a = -0.5 * dt / (2 * dx**2)  # off-diagonal
b = 1j + dt / dx**2        # diagonal (kinetic term)
A = np.zeros((3, N+1), dtype=complex)
A[0, 1:] = a               # upper diagonal
A[1, :] = b                # main diagonal
A[2, :-1] = a              # lower diagonal
A[1, 0], A[1, -1] = 1, 1   # BCs

# Time evolution
psi_t = [psi.copy()]
for n in range(M):
    t = n * dt
    Vn = V(x, t)
    rhs = psi + 0.5 * dt * (Vn + g * np.abs(psi)**2) * psi  # Explicit part
    rhs[0], rhs[-1] = 0, 0  # BCs
    psi = solve_banded((1, 1), A, rhs)
    psi_t.append(psi.copy())

# Plot
for n in [0, M//2, M]:
    plt.plot(x, np.abs(psi_t[n])**2, label=f't={n*dt:.2f}')
plt.legend()
plt.show()