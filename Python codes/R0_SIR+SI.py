
"""
 SIR-SI Model: IVP
     
          # Human Population #  
dSh/dt = Lambdah - betah * Sh * Iv/Nh - muh * Sh
dIh/dt = betah * Sh * Ih - gammah * Ih - muh * Ih
dRh/dt = gammah * Ih - muh * Rh

            # Vector Population #
            
dSv/dt = Lambdav - betav *Ih*Sv/Nh-muv*Sv
dIv/dt = betav *Ih*Sv/Nh - muv * Iv
        

"""
import sympy as sp
# Define symbols
Sh, Ih, Rh,Sv, Iv = sp.symbols('S_h I_h R_h S_v I_v')
Nh, Nv=sp.symbols('N_h N_v')
Nh = Sh+Ih+Rh
betah, betav, gammah, muh, muv, Lambdah, Lambdav = sp.symbols('beta_h beta_v gamma_h mu_h mu_v Lambda_h Lambda_v')
# Infected compartment equation
dIh_dt = betah * Sh * Iv/Nh  - gammah * Ih - muh * Ih
dIv_dt = betav* Sv *Ih/Nh - muv* Iv
# Step 1: Compute F (new infections) and V (transitions)
F = sp.Matrix([betah * Sh * Iv/Nh, betav*Sv*Ih/Nh] )  # New infections
V = sp.Matrix([(gammah + muh) * Ih, muv*Iv])  # Outflow from I (recovery + death)
sp.pretty_print(F)
sp.pprint(V)
# Step 2: Compute Jacobians of F and V w.r.t. [E, I]
variables = [Ih, Iv]  # Differentiate with respect to E and I
# Jacobian of F (2x2 matrix)
F_jacobian = sp.Matrix([
    [sp.diff(F[0], Ih), sp.diff(F[0], Iv)],  # ∂F₁/∂Ih, ∂F₁/∂Iv
    [sp.diff(F[1], Ih), sp.diff(F[1], Iv)]   # ∂F₂/∂Ih, ∂F₂/∂Iv
])
print("\nJacobian of F:")
sp.pprint(F_jacobian)
# Jacobian of V (2x2 matrix)
V_jacobian = sp.Matrix([
    [sp.diff(V[0], Ih), sp.diff(V[0], Iv)],  # ∂V₁/∂E, ∂V₁/∂I
    [sp.diff(V[1], Ih), sp.diff(V[1], Iv)]   # ∂V₂/∂E, ∂V₂/∂I
])
print("\nJacobian of V:")
sp.pprint(V_jacobian)
# Step 3: Evaluate at Disease-Free Equilibrium (DFE)
DFE = {Sh: Lambdah/muh, Ih: 0, Rh: 0, Sv: Lambdav/muv , Iv: 0}
#print("\nDisease-Free Equilibrium (DFE):")
#sp.pprint((DFE[Sh], DFE[Ih], DFE[Rh], DFE[Sv], DFE[Iv]))
F=F_at_DFE = F_jacobian.subs(DFE)
V=V_at_DFE = V_jacobian.subs(DFE)
# print("\nThe Jacobian of the matrix:\n \n F=\n ")
# sp.pprint(F)
# Step 4: Compute Next Generation Matrix (K = FV⁻¹)
param_values = {
    Lambdah: 1,    # Recruitment rate
    Lambdav : 10,  #Recruitment rate of vector
    betah: 0.001,     # Transmission rate from vector to human
    betav : 0.0001, #Transmission rate from human to vector
    muh: 0.005,     # Natural death rate
    muv : 0.0001, # Natural death rate of vector
    gammah: 0.01,  # Recovery rate   
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
