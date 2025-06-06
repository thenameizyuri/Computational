# -*- coding: utf-8 -*-
"""
Compare: 
1) Neural Network with 2 hidden neurons
2) Analytical sigmoid model with 2 hidden neurons (from your earlier optimization)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from io import StringIO

# ===== Load Data =====
data_raw = """
Temp	Dengue_outbreak
10.83	0
10.84	0
11.23	0
11.24	0
11.44	0
11.54	0
11.69	0
11.86	0
11.89	0
11.96	0
12.14	0
12.29	0
12.31	0
13.19	0
13.36	0
13.44	0
13.66	0
13.91	0
14.44	0
14.84	0
15.24	0
15.99	0
16.01	0
16.11	0
16.81	0
17.19	1
17.24	1
17.89	1
18.91	1
19.3	1
19.89	1
20.34	1
20.66	1
21.44	1
22.16	1
22.36	1
22.39	1
22.44	1
22.56	1
22.74	1
22.86	0
22.89	0
23.09	0
23.31	0
23.46	0
23.61	0
23.89	0
23.91	0
24.19	0
24.19	0
24.34	0
24.36	0
"""
df = pd.read_csv(StringIO(data_raw), sep="\t")
x_data = df["Temp"].values.reshape(-1,1)
y_data = df["Dengue_outbreak"].values.reshape(-1,1)

# Normalize input for NN
X_mean, X_std = x_data.mean(), x_data.std()
X_norm = (x_data - X_mean) / X_std

# ======= 1. Neural Network with 2 hidden neurons =======

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(y):
    return y * (1 - y)

np.random.seed(42)
n_hidden = 2

# Initialize weights
w1 = np.random.randn(1, n_hidden)
b1 = np.zeros((1, n_hidden))
w2 = np.random.randn(n_hidden, 1)
b2 = np.zeros((1, 1))

lr = 0.1
epochs = 20000
losses_nn = []

X = X_norm
Y = y_data

for epoch in range(epochs):
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    y_pred = sigmoid(z2)
    
    # Binary cross-entropy loss
    epsilon = 1e-8
    loss = -np.mean(Y * np.log(y_pred + epsilon) + (1 - Y) * np.log(1 - y_pred + epsilon))
    losses_nn.append(loss)
    
    # Backpropagation
    d_loss = y_pred - Y
    d_w2 = np.dot(a1.T, d_loss)
    d_b2 = np.sum(d_loss, axis=0, keepdims=True)

    d_a1 = np.dot(d_loss, w2.T)
    d_z1 = d_a1 * sigmoid_derivative(a1)
    d_w1 = np.dot(X.T, d_z1)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)
    
    # Update weights
    w2 -= lr * d_w2
    b2 -= lr * d_b2
    w1 -= lr * d_w1
    b1 -= lr * d_b1

# Predictions NN on full range
temps_test = np.linspace(x_data.min() - 1, x_data.max() + 1, 300).reshape(-1,1)
X_test = (temps_test - X_mean) / X_std
a1_test = sigmoid(np.dot(X_test, w1) + b1)
y_test_pred_nn = sigmoid(np.dot(a1_test, w2) + b2)

# ======= 2. Analytical 2-neuron sigmoid model with parameters from your earlier code =======

# Model function for analytical sigmoid model with 2 hidden neurons
def model_output(x, params):
    w11, b1a, w12, b2a, w21, w22, b3 = params
    h1 = 1 / (1 + np.exp(-(w11 * x + b1a)))
    h2 = 1 / (1 + np.exp(-(w12 * x + b2a)))
    z = w21 * h1 + w22 * h2 + b3
    y = 1 / (1 + np.exp(-z))
    return y

# Loss function
def loss_fn(params):
    y_pred = model_output(x_data.flatten(), params)
    return np.sum((y_pred - y_data.flatten())**2)

# Initial guess (tuned to your earlier example)
initial_params = [5.0, -75.0, -5.0, 95.0, 6.0, 6.0, -6.0]
result = minimize(loss_fn, initial_params, method='L-BFGS-B')

optimized_params = result.x

# Predictions analytical model
y_test_pred_analytical = model_output(temps_test.flatten(), optimized_params)

# ======= Plotting both results =======
plt.figure(figsize=(14, 6))

plt.subplot(1,2,1)
plt.scatter(x_data, y_data, color='red', label='True Data', s=50)
plt.plot(temps_test, y_test_pred_nn, color='blue', linewidth=3, label='Neural Network (2 neurons)')
plt.xlabel('Temperature (°C)')
plt.ylabel('Outbreak Occurrence')
plt.title('Neural Network Prediction (2 hidden neurons)')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.scatter(x_data, y_data, color='red', label='True Data', s=50)
plt.plot(temps_test, y_test_pred_analytical, color='green', linewidth=3, label='Analytical Sigmoid Model')
plt.xlabel('Temperature (°C)')
plt.ylabel('Outbreak Occurrence')
plt.title('Analytical Sigmoid Model (2 neurons)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print weights and biases from the first (manual sigmoid) model
print("\n=== Model 1: Manual Sigmoid with 2 Neurons ===")
param_names = ['w11', 'b1', 'w12', 'b2', 'w21', 'w22', 'b3']
for name, val in zip(param_names, optimized_params):
    print(f"{name} = {val:.6f}")

# Print weights and biases from the neural network model
print("\n=== Model 2: Neural Network with 2 Neurons ===")

print("\nWeights Input to Hidden (w1):")
for i in range(w1.shape[1]):
    print(f"w1[0][{i}] = {w1[0][i]:.6f}")

print("\nBiases Hidden Layer (b1):")
for i in range(b1.shape[1]):
    print(f"b1[0][{i}] = {b1[0][i]:.6f}")

print("\nWeights Hidden to Output (w2):")
for i in range(w2.shape[0]):
    print(f"w2[{i}][0] = {w2[i][0]:.6f}")

print("\nBias Output Layer (b2):")
print(f"b2[0][0] = {b2[0][0]:.6f}")
