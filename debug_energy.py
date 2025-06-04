#!/usr/bin/env python3

import numpy as np
import sys
sys.path.append('src')
from warp_qft.negative_energy import compute_energy_density, integrate_negative_energy_over_time

def test_energy_difference():
    """Test whether polymer energy can be lower than classical."""
    
    print("Testing kinetic energy differences...")
    mu = 0.3
    # Test pi values around regions where sin(mu*pi) should give lower energy
    pi_values = np.array([0.0, 2.8, 3.3, 4.0, 5.2, 5.7, 6.0])
    
    classical_kinetic = pi_values**2 / 2
    polymer_kinetic = (np.sin(mu * pi_values) / mu)**2 / 2
    difference = polymer_kinetic - classical_kinetic
    
    print(f'π values: {pi_values}')
    print(f'μπ values: {mu * pi_values}')
    print(f'sin(μπ): {np.sin(mu * pi_values)}')
    print(f'Classical kinetic: {classical_kinetic}')
    print(f'Polymer kinetic: {polymer_kinetic}')
    print(f'Difference (polymer - classical): {difference}')
    print(f'Negative differences: {difference[difference < 0]}')
    
    if len(difference[difference < 0]) > 0:
        print("✓ Found negative energy differences!")
    else:
        print("✗ No negative energy differences found")
    
    print("\nTesting full integration...")
    N = 32
    dx = 1.0
    dt = 0.02
    total_time = 8.0
    tau = 1.0
    
    for mu_test in [0.3, 0.6]:
        I_diff = integrate_negative_energy_over_time(N, mu_test, total_time, dt, dx, tau)
        print(f"μ={mu_test}: I_polymer - I_classical = {I_diff}")
        
        # Let's also check some intermediate values        # Set up the same calculation manually
        x = np.arange(N) * dx
        x0 = N * dx / 2
        sigma = N * dx / 8
        A = 6.0 / mu_test  # Updated to match new function
        omega = 2 * np.pi / total_time
        
        times = np.linspace(-total_time/2, total_time/2, 10)
        
        print(f"  Parameters: A={A:.3f}, ω={omega:.3f}")
        print(f"  Peak μπ value: {mu_test * A:.3f}")
        
        for i, t in enumerate(times[:3]):  # Just first few times
            envelope = np.exp(-((x - x0)**2) / (2 * sigma**2))
            pi_t = A * envelope * np.sin(omega * t)
            phi_t = np.zeros_like(pi_t)
            
            rho_polymer = compute_energy_density(phi_t, pi_t, mu_test, dx)
            rho_classical = compute_energy_density(phi_t, pi_t, 0.0, dx)
            
            print(f"    t={t:.2f}: max(π)={np.max(pi_t):.3f}, max(ρ_poly)={np.max(rho_polymer):.3f}, max(ρ_class)={np.max(rho_classical):.3f}")

if __name__ == "__main__":
    test_energy_difference()
