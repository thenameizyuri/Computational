from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M = 1.989e30     # Mass of the black hole (kg)
c = 3.0e8        # Speed of light (m/s)
r_s = 2 * G * M / c**2  # Schwarzschild radius

# Differential equations for radial and angular motion (function banako)
def equations(t, y):
    r, phi, pr, pphi = y
    dr_dt = pr
    dphi_dt = pphi / r**2
    dpr_dt = -G * M / r**2 + pphi**2 / r**3
    dpphi_dt = 0
    return [dr_dt, dphi_dt, dpr_dt, dpphi_dt]

# Initial conditions
r0 = 10 * r_s  # Initial radius, 10 times the Schwarzschild radius
phi0 = 0       # Initial angle
pr0 = 0        # Initial radial momentum
pphi0 = 4.0    # Initial angular momentum

# Solving the equations
sol = solve_ivp(equations, [0, 100], [r0, phi0, pr0, pphi0], max_step=0.1)

# Converting to Cartesian coordinates for plotting(spherical polar maa ali garoww huncha)
x = sol.y[0] * np.cos(sol.y[1])
y = sol.y[0] * np.sin(sol.y[1])

# Plotting the orbit
plt.figure(figsize=(8, 8))
plt.plot(x, y, label='Orbit around black hole')
plt.scatter(0, 0, color='red', label='Black Hole', s=100)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.title('Particle Orbit in Schwarzschild Geometry')
plt.grid()
plt.show()
