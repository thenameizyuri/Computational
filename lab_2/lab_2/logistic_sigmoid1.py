import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

# ========== Load Data ==========
data = pd.read_csv("lab_2/lab_2/Temp_Infection_01.csv")
x_data = data.iloc[:, 0].values  # Temperature
y_data = data.iloc[:, 1].values  # Infection labels (0 or 1)

# ========== Numerically Stable Sigmoid Function ==========
def sigmoid(x, beta0, beta1):
    z = beta1 + beta0 * x
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

# ========== Mean Squared Error Loss Function ==========
def mse_loss(params, x, y_true):
    beta0, beta1 = params
    y_pred = sigmoid(x, beta0, beta1)
    return np.mean((y_true - y_pred) ** 2)

# ========== Optimization ==========
initial_guess = [0.0, 1.0]
result = minimize(mse_loss, initial_guess, args=(x_data, y_data))

# ========== Fitted Parameters ==========
beta0, beta1 = result.x
print(f"Fitted coefficients:\nbeta0 = {beta0:.4f}, beta1 = {beta1:.4f}")

# ========== Final Predictions and MSE ==========
y_pred = sigmoid(x_data, beta0, beta1)
mse = np.mean((y_data - y_pred) ** 2)
print(f"Mean Squared Error (MSE): {mse:.6f}")

# ========== Plot Fitted Curve ==========
x_fit = np.linspace(min(x_data), max(x_data), 200)
y_fit = sigmoid(x_fit, beta0, beta1)

plt.figure(figsize=(8, 5))
plt.scatter(x_data, y_data, color='blue', label='Observed Data')
plt.plot(x_fit, y_fit, color='red', label='Fitted Sigmoid Curve')
plt.xlabel('Temperature (°C)')
plt.ylabel('Outbreak Occurrence Probability')
plt.title('Sigmoid Fit with Numerically Stable Function')
plt.legend(loc='center right')
plt.grid(True)
plt.show()


# Prediction for a specific temperature
temp_check = 16
pred_prob = sigmoid(temp_check, beta0, beta1)
print(f"Predicted probability at {temp_check}°C: {pred_prob:.4f}")
