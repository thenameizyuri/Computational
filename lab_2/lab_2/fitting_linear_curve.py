import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("lab_2/lab_2/Temp_linear.csv")
T = data.iloc[:, 0].values  # Temperature (x)
I = data.iloc[:, 1].values  # Dengue cases (y)

# Prepare X matrix: [T, 1] for intercept term
X = np.column_stack((T, np.ones_like(T)))

# Least squares solution: theta = (X^T X)^(-1) X^T I
theta = np.linalg.inv(X.T @ X) @ X.T @ I
a, b = theta  # slope (a), intercept (b)

# Predictions
I_pred = a * T + b

# Print results
print(f"Slope (a): {a:.4f}")
print(f"Intercept (b): {b:.4f}")
print(f"Equation: Dengue Cases = {a:.2f} × Temperature + {b:.2f}")

# Plot
plt.scatter(T, I, color='blue', label='Actual Data')
plt.plot(T, I_pred, color='red', label='Linear Prediction')
plt.xlabel('Temperature (°C)', fontsize=14)
plt.ylabel('Dengue Cases')
plt.legend()
plt.grid(True)
plt.title('Least Squares Linear Regression')
plt.show()

# Ask for user input and print predicted outcome
# T_input = float(input("\nEnter a Temperature (°C) to predict dengue cases: "))
# I_output = a * T_input + b
# print(f"Predicted Dengue Cases at {T_input:.2f}°C: {I_output:.2f}")
