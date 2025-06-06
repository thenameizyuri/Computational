# -*- coding: utf-8 -*-
"""
Neural network: Temperature → Infection likelihood (binary)
1 input → 10 hidden neurons → 1 output
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================== Load Data ===================
data = pd.read_csv("D:\\Python\\AESIM2025\\Neural network\\Temp_Infection_010.csv")
temps = data.iloc[:, 0].values.reshape(-1, 1)
labels = data.iloc[:, 1].values.reshape(-1, 1)

# =================== Normalize Input ===================
X_mean, X_std = temps.mean(), temps.std()
X = (temps - X_mean) / X_std
Y = labels

# =================== Sigmoid Functions ===================
def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(y): return y * (1 - y)

# =================== Initialize Weights ===================
np.random.seed(42)
n_hidden = 10
w1 = np.random.randn(1, n_hidden)
b1 = np.zeros((1, n_hidden))
w2 = np.random.randn(n_hidden, 1)
b2 = np.zeros((1, 1))

# =================== Training Parameters ===================
lr = 0.1
epochs = 20000
losses = []

# =================== Training Loop ===================
for epoch in range(epochs):
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    y_pred = sigmoid(z2)

    # Binary cross-entropy loss
    epsilon = 1e-8
    loss = -np.mean(Y * np.log(y_pred + epsilon) + (1 - Y) * np.log(1 - y_pred + epsilon))
    losses.append(loss)

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

# =================== Print Weights and Biases ===================
print("\n=== Final Weights and Biases ===\n")

print("Weights from Input to Hidden Layer (w1):")
for i in range(n_hidden):
    print(f"w1[0][{i}] = {w1[0][i]:.6f}")
    
print("\nBiases of Hidden Layer (b1):")
for i in range(n_hidden):
    print(f"b1[0][{i}] = {b1[0][i]:.6f}")

print("\nWeights from Hidden to Output Layer (w2):")
for i in range(n_hidden):
    print(f"w2[{i}][0] = {w2[i][0]:.6f}")

print("\nBias of Output Layer (b2):")
print(f"b2[0][0] = {b2[0][0]:.6f}")

# =================== Visualization ===================
temps_test = np.linspace(temps.min() - 1, temps.max() + 1, 300).reshape(-1, 1)
X_test = (temps_test - X_mean) / X_std
a1_test = sigmoid(np.dot(X_test, w1) + b1)
y_test_pred = sigmoid(np.dot(a1_test, w2) + b2)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(temps, labels, color='red', label='True Data', s=50)
plt.plot(temps_test, y_test_pred, color='blue', label='NN Prediction', linewidth=3)
plt.xlabel('Temperature (°C)')
plt.ylabel('Infection Likelihood')
plt.legend()
plt.grid(True)
plt.title('Fit: Temperature vs. Predicted Infection')

plt.subplot(1, 2, 2)
plt.plot(losses, color='purple')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)

plt.tight_layout()
plt.show()

# =================== Predict for New Input ===================
# T = float(input("Enter a Temperature value T: "))
# T_norm = (T - X_mean) / X_std
# a1_input = sigmoid(np.dot([[T_norm]], w1) + b1)
# y_input_pred = sigmoid(np.dot(a1_input, w2) + b2)
# print(f"\nPredicted Infection Likelihood at T = {T}°C: {y_input_pred[0][0]:.6f}")
