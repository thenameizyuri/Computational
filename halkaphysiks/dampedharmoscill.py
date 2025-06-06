import numpy as np
import matplotlib.pyplot as plt

# Parameters
omega = 2.0  # Natural frequency
gamma = 1  # Damping coefficient
t_max = 100.0  # Total time
dt = 0.01  # Time step

# Initial conditions
x0 = 1.0  # Initial displacement
v0 = 0.0  # Initial velocity

# Time array
t = np.arange(0, t_max, dt)
x = np.zeros_like(t)
v = np.zeros_like(t)

# setting initial values
x[0], v[0] = x0, v0

# ahile lai Euler method for numerical integration
for i in range(1, len(t)):
    a = -2 * gamma * v[i-1] - omega**2 * x[i-1]
    v[i] = v[i-1] + a * dt
    x[i] = x[i-1] + v[i-1] * dt

# Plotting the results
plt.plot(t, x, label='Displacement (x)')
plt.title('Damped Harmonic Oscillator')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()
plt.show()
