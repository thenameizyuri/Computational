import numpy as np
import matplotlib.pyplot as plt


L = 20 
T = 2.5 
J = 1.0  
num_steps = 10000

# Initialize a random spin configuration (-1 or +1)
lattice = np.random.choice([-1, 1], size=(L, L))

def energy(lattice):
    E = 0
    for i in range(L):
        for j in range(L):
            S = lattice[i, j]
            neighbors = lattice[(i+1)%L, j] + lattice[i, (j+1)%L] + lattice[(i-1)%L, j] + lattice[i, (j-1)%L]
            E += -J * S * neighbors
    return E / 2  

def monte_carlo_step(lattice, T):
    
    for _ in range(L**2): 
        i, j = np.random.randint(0, L, 2) 
        S = lattice[i, j]
        neighbors = lattice[(i+1)%L, j] + lattice[i, (j+1)%L] + lattice[(i-1)%L, j] + lattice[i, (j-1)%L]
        dE = 2 * J * S * neighbors

        # Metropolis acceptance criterion
        if dE < 0 or np.random.rand() < np.exp(-dE / T):
            lattice[i, j] *= -1  # Flip spin

energies = []
for step in range(num_steps):
    monte_carlo_step(lattice, T)
    if step % 1000 == 0:  
        energies.append(energy(lattice))

plt.plot(energies)
plt.xlabel("Monte Carlo Steps (x1000)")
plt.ylabel("Energy")
plt.title("Energy vs Monte Carlo Steps (Ising Model)")
plt.show()

plt.imshow(lattice, cmap='gray')
plt.title("Final Spin Configuration")
plt.show()

