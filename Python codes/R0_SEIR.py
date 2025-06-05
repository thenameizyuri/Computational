# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 00:52:11 2025

@author: USER
"""

"""
    SEIR Model Equations: IVP

#ODE 
  
dS/dt = Lambda - beta * S * I - mu * S
dE/dt =  beta * S * I - lambda * E - mu * E
dI/dt = lambda * E - gamma * I - mu * I
dR/dt = gamma * I - mu * R

#IV 

S(0) = S0, E(0) = E0, I(0) =  I0, R(0) = R0 

"""
import sympy as sp

# Define symbols (with positive=True for realistic parameters)
S, E, I, R = sp.symbols('S E I R')
beta, gamma, mu, Lambda, lamda = sp.symbols('beta gamma mu Lambda lambda', positive=True)
#N = S + E + I + R  # Total population (optional for frequency-dependent transmission)
#Infected equations
dE_dt =  beta * S * I - lamda * E - mu * E
dI_dt = lamda * E - gamma * I - mu * I
# New infections matrix (F)
F = sp.Matrix([
    beta * S * I ,  # New infections entering E
    0                  # No direct infections in I
])

# Transitions matrix (V)
V = sp.Matrix([
    (lamda + mu) * E,             # Outflow from E (becoming infectious or dying)
    -lamda * E + (gamma + mu) * I  # Outflow from I (recovery or death)
])

print("New Infections Matrix (F):")
sp.pprint(F)
print("\nTransitions Matrix (V):")
sp.pprint(V)

# Step 2: Compute Jacobians of F and V w.r.t. [E, I]
variables = [E, I]  # Differentiate with respect to E and I

# Jacobian of F (2x2 matrix)
F_jacobian = sp.Matrix([
    [sp.diff(F[0], E), sp.diff(F[0], I)],  # ∂F₁/∂E, ∂F₁/∂I
    [sp.diff(F[1], E), sp.diff(F[1], I)]   # ∂F₂/∂E, ∂F₂/∂I
])

# Jacobian of V (2x2 matrix)
V_jacobian = sp.Matrix([
    [sp.diff(V[0], E), sp.diff(V[0], I)],  # ∂V₁/∂E, ∂V₁/∂I
    [sp.diff(V[1], E), sp.diff(V[1], I)]   # ∂V₂/∂E, ∂V₂/∂I
])

print("\nJacobian of F:")
sp.pprint(F_jacobian)
print("\nJacobian of V:")
sp.pprint(V_jacobian)

# Step 3: Evaluate at Disease-Free Equilibrium (DFE)
DFE = {S: Lambda/mu, E: 0, I: 0, R: 0}
#print("\nDisease-Free Equilibrium (DFE):")
#sp.pprint((DFE[S], DFE[E], DFE[I], DFE[R]))

F_at_DFE = F_jacobian.subs(DFE)
V_at_DFE = V_jacobian.subs(DFE)

# print("\nF evaluated at DFE:")
# sp.pprint(F_at_DFE)
# print("\nV evaluated at DFE:")
# sp.pprint(V_at_DFE)
# Substitute numerical values for parameters
param_values = {
    Lambda: 1,    # Recruitment rate
    beta: 0.001,     # Transmission rate
    mu: 0.005,     # Natural death rate
    gamma: 0.01,  # Recovery rate
    lamda : 0.01 # Infectious Rate
}
# Substitute into F and V at DFE
F = F_at_DFE.subs(param_values)
V = V_at_DFE.subs(param_values)
# Step 4: Compute Next Generation Matrix (K = FV⁻¹)
V_inv = V.inv()
NGM = F * V_inv
print("\nNext Generation Matrix (K = FV⁻¹):")
sp.pprint(NGM)
eigenvals = NGM.eigenvals()
eigenvalues = eigenvals.keys()
# Step 5: Calculate R₀ (spectral radius of K)
R0 = max(eigenvalues)
R0 = sp.simplify(R0)
print("\nBasic Reproduction Number R0 =",R0)
#sp.pprint(R0)
