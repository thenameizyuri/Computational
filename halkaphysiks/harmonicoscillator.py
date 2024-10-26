import numpy as np
import matplotlib.pyplot as plt

# Parameters
omega = 1.0  # Angular frequency
t_max = 10.0  # Total time
dt = 0.01  # Time step

# Initial conditions
x0 = 1.0  # Initial displacement
v0 = 0.0  # Initial velocity

# Time array
t = np.arange(0, t_max, dt)
x = np.zeros_like(t)
v = np.zeros_like(t)

# Initial values
x[0], v[0] = x0, v0

# Runge-Kutta method iteratiion solving diffeqns
for i in range(1, len(t)):
    k1_x = v[i-1]
    k1_v = -omega**2 * x[i-1]

    k2_x = v[i-1] + 0.5 * dt * k1_v
    k2_v = -omega**2 * (x[i-1] + 0.5 * dt * k1_x)

    k3_x = v[i-1] + 0.5 * dt * k2_v
    k3_v = -omega**2 * (x[i-1] + 0.5 * dt * k2_x)

    k4_x = v[i-1] + dt * k3_v
    k4_v = -omega**2 * (x[i-1] + dt * k3_x)

    x[i] = x[i-1] + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    v[i] = v[i-1] + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

# Plotting the results
plt.plot(t, x, label='Displacement (x)')
plt.plot(t, v, label='Velocity (v)', linestyle='--')
plt.title('Simple Harmonic Oscillator')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
