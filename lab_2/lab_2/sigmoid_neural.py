# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 09:39:52 2025

@author: USER
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from graphviz import Digraph

# Load data
data = pd.read_csv("lab_2/lab_2/Temp_Infection_01.csv")
x = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1].values.reshape(-1, 1)

# Normalize input
x_min, x_max = x.min(), x.max()
x_norm = (x - x_min) / (x_max - x_min)

# Sigmoid activation and derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

# Initialize one weight and one bias (1-1-1 architecture)
np.random.seed(0)
w1 = np.random.randn(1, 1)     # input to single neuron
b1 = np.zeros((1, 1))          # bias for single neuron

# Training
lr = 0.3
epochs = 10000
for epoch in range(epochs):
    # Forward pass
    z1 = x_norm @ w1 + b1
    y_pred = sigmoid(z1)

    # Loss (MSE)
    loss = np.mean((y_pred - y) ** 2)

    # Backward pass
    dz = (y_pred - y) * sigmoid_deriv(z1)
    dw1 = x_norm.T @ dz
    db1 = np.sum(dz, axis=0, keepdims=True)

    # Update weights
    w1 -= lr * dw1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Plot fitted curve
x_fit = np.linspace(0, 1, 100).reshape(-1, 1)
y_fit = sigmoid(x_fit @ w1 + b1)
x_plot = x_fit * (x_max - x_min) + x_min

plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='red', label='Data', s=60)  # Bigger data points
plt.plot(x_plot, y_fit, color='blue', label='Neural Network Output', linewidth=3)  # Thicker fitted curve
plt.xlabel('Temperature (Â°C)', fontsize=14)
plt.ylabel('Outbreak Occurrence', fontsize=14)
#plt.title('1-1-1 Neural Network Fit', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()



# Print final parameters and output
print("\n--- Learned Parameters ---")
print(f"Weight w1: {w1[0][0]:.4f}")
print(f"Bias b1: {b1[0][0]:.4f}")
print(f"Final prediction on last input: {sigmoid(x_norm[-1:] @ w1 + b1)[0][0]:.4f}")

#from graphviz import Digraph

# def plot_io_nn():
#     dot = Digraph(format='png')
#     dot.attr(rankdir='LR', size='8,5')  # Wider layout

#     # Input neuron
#     dot.node('Input', shape='circle', style='filled', color='lightblue', label='Input\n(Temp Â°C)')

#     # Output neuron
#     dot.node('Output', shape='circle', style='filled', color='salmon', label='Output\n(Occurrence)')

#     # Direct connection from input to output
#     dot.edge('Input', 'Output', label='f(x)')

#     # Legend (optional)
#     with dot.subgraph(name='cluster_legend') as c:
#         c.attr(label='Legend', color='gray')
#         c.node('L1', label='Input Neuron', shape='circle', style='filled', color='lightblue')
#         c.node('L2', label='Output Neuron', shape='circle', style='filled', color='salmon')
#         c.edge('L1', 'L2', label='f(x)', style='solid')

#     return dot

# # Render the updated diagram
# simple_nn = plot_io_nn()
# simple_nn.render('simple_input_output_nn', view=True)
