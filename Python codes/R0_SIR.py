"""
SIR Model Equations: IVP

    #ODE
    
    dS/dt = Lambda - beta * S * I - mu * S
    dI/dt = beta * S * I - gamma * I - mu * I
    dR/dt = gamma * I - mu * R
    
   # IV: S(0) = S0, I(0) = I0, R(0) = R0
    
"""
import sympy as sp

# Define symbols
S, I, R = sp.symbols('S I R')
beta, gamma, mu, Lambda = sp.symbols('beta gamma mu Lambda')

# Infected compartment equation
dI_dt = beta * S * I - gamma * I - mu * I

# Step 1: Compute F (new infections) and V (transitions)
F = beta * S * I   # New infections
V = (gamma + mu) * I  # Outflow from I (recovery + death)

# Step 2: Compute Jacobians of F and V w.r.t. I
F_jacobian = sp.diff(F, I)
V_jacobian = sp.diff(V, I)

# Step 3: Evaluate at Disease-Free Equilibrium (DFE)
DFE = {S: Lambda/mu, I: 0, R: 0}
#print("Disease-Free Equilibrium (DFE):", DFE)

# Substitute numerical values for parameters
param_values = {
    Lambda: 1,    # Recruitment rate
    beta: 0.001,     # Transmission rate
    mu: 0.005,     # Natural death rate
    gamma: 0.01  # Recovery rate
}

# Substitute into F and V at DFE
F_at_DFE = F_jacobian.subs(DFE).subs(param_values) #1*1 matrix
V_at_DFE = V_jacobian.subs(DFE).subs(param_values) #1*1 matrix

# Compute R0 numerically
R0 = F_at_DFE / V_at_DFE

print("Basic Reproduction Number R0 is")
print(R0)  # .evalf() ensures a floating-point result