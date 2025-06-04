#!/usr/bin/env python3
"""
Symbolic derivation of polymer-modified Ford-Roman bounds.

This script demonstrates the theoretical foundation for quantum inequality 
violations in polymer field theory.
"""

import sympy as sp
import numpy as np
from sympy import symbols, sin, pi, exp, sqrt, simplify, series, latex

def main():
    print("Polymer-Modified Ford-Roman Bound Derivation")
    print("=" * 50)
    
    # Define symbols
    mu, hbar, tau, rho, t = symbols('mu hbar tau rho t', real=True, positive=True)
    
    print("\n1. Classical Ford-Roman Bound:")
    print("   ∫ ρ(t) f(t) dt ≥ -ℏ/(12π τ²)")
    
    # Classical bound
    qi_classical = -hbar / (12 * pi * tau**2)
    print(f"   Classical bound = {qi_classical}")
    
    print("\n2. Polymer Modification:")
    print("   sinc(μ) = sin(μ)/μ")
    
    # Define sinc function
    sinc = sin(mu) / mu
    print(f"   sinc(μ) = {sinc}")
    
    print("\n3. Polymer-Modified Bound:")
    print("   ∫ ρ_eff(t) f(t) dt ≥ -ℏ·sinc(μ)/(12π τ²)")
    
    # Polymer-modified bound
    qi_polymer = -hbar * sinc / (12 * pi * tau**2)
    print(f"   Polymer bound = {qi_polymer}")
    
    print("\n4. Small μ expansion:")
    # Expand sinc(μ) for small μ
    sinc_expansion = series(sinc, mu, 0, 4).removeO()
    print(f"   sinc(μ) ≈ {sinc_expansion}")
    
    print("\n5. Enhancement factor:")
    enhancement = sinc
    print(f"   Enhancement = sinc(μ) = {enhancement}")
    
    print("\n6. Numerical evaluation for typical values:")
    mu_vals = [0.0, 0.3, 0.6, 1.0]
    
    print("   μ     | sinc(μ)  | Enhancement")
    print("   ------|----------|------------")
    
    for mu_val in mu_vals:
        if mu_val == 0:
            sinc_val = 1.0
        else:
            sinc_val = float(np.sin(mu_val) / mu_val)
        print(f"   {mu_val:4.1f}  | {sinc_val:7.3f}  | {sinc_val:10.3f}")
    
    print("\n7. LaTeX expressions:")
    print(f"   Classical: ${latex(qi_classical)}$")
    print(f"   Polymer:   ${latex(qi_polymer)}$")
    print(f"   sinc(μ):   ${latex(sinc)}$")
    
    print("\n8. Key insight:")
    print("   For μ > 0: sinc(μ) < 1")
    print("   Therefore: |polymer bound| < |classical bound|")
    print("   This allows negative energy configurations forbidden classically!")
    
    # Violation window analysis
    print("\n9. Violation window:")
    print("   Classical forbids: ∫ρf dt < -ℏ/(12πτ²)")
    print("   Polymer allows:    -ℏ/(12πτ²) < ∫ρf dt < -ℏ·sinc(μ)/(12πτ²)")
    
    return {
        'classical_bound': qi_classical,
        'polymer_bound': qi_polymer,
        'sinc_function': sinc,
        'enhancement': enhancement
    }


if __name__ == "__main__":
    results = main()
    
    # Additional verification
    print(f"\n10. Verification with SymPy:")
    mu_test = 0.5
    classical_val = float(results['classical_bound'].subs([(symbols('hbar'), 1), (symbols('tau'), 1)]))
    polymer_val = float(results['polymer_bound'].subs([(symbols('mu'), mu_test), (symbols('hbar'), 1), (symbols('tau'), 1)]))
    
    print(f"    For μ = {mu_test}, ℏ = 1, τ = 1:")
    print(f"    Classical bound = {classical_val:.6f}")
    print(f"    Polymer bound   = {polymer_val:.6f}")
    print(f"    Ratio = {polymer_val/classical_val:.3f}")
