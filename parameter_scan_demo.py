#!/usr/bin/env python3
"""
Parameter scan for warp bubble feasibility optimization.

This script implements the parameter scanning suggestions from the user's request,
plotting feasibility ratio vs μ and R to identify optimal configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def gaussian_sampling(t, tau):
    """Normalized Gaussian sampling function f(t,τ)."""
    return np.exp(-t**2/(2*tau**2)) / (np.sqrt(2*np.pi)*tau)

def toy_negative_energy_density(x, mu, R, rho0=1.0, sigma=None):
    """
    Toy model of a negative‐energy distribution inside radius R:
    ρ(x) = -ρ0 * exp[-(x/σ)^2] * sinc(mu).
    """
    if sigma is None:
        sigma = R / 2
    return -rho0 * np.exp(-(x**2)/(sigma**2)) * np.sinc(mu)

def available_negative_energy(mu, tau, R, Nx=200, Nt=200):
    """
    Compute total negative energy by integrating ρ(x)*f(t) over x∈[-R,R] and t∈[-5τ,5τ].
    """
    x = np.linspace(-R, R, Nx)
    t = np.linspace(-5*tau, 5*tau, Nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Precompute sampling function and spatial profile
    f_t = gaussian_sampling(t, tau)           # shape: (Nt,)
    rho_x = toy_negative_energy_density(x, mu, R)  # shape: (Nx,)

    # Total energy = ∫ (ρ(x) dx) * (∫ f(t) dt)
    total_rho = np.sum(rho_x) * dx            # ∫ ρ(x) dx
    total_f = np.sum(f_t) * dt                # ∫ f(t) dt (≈1 by normalization)
    return total_rho * total_f   # scalar

def warp_energy_requirement(R, v=1.0, c=1.0):
    """
    Rough estimate of energy required to form a warp bubble of radius R at speed v:
    E_req ≈ α * R * v^2, with α ~ O(1) in Planck units.
    """
    α = 1.0  # dimensionless prefactor
    return α * R * (v**2) / (c**2)

def parameter_scan():
    """
    Automate a parameter scan over μ, τ, and R to identify the smallest R 
    for which E_avail/E_req ≳ 1.
    """
    print("🔍 PARAMETER SCAN FOR WARP BUBBLE FEASIBILITY")
    print("=" * 50)
    
    # Parameter ranges
    mu_vals = np.linspace(0.1, 0.6, 20)
    R_vals = np.linspace(0.5, 3.0, 25)
    tau = 1.0  # Fixed for this scan
    v = 1.0    # Fixed velocity
    
    # Storage for results
    feasibility_grid = np.zeros((len(mu_vals), len(R_vals)))
    best_ratio = 0
    best_config = None
    feasible_configs = []
    
    print(f"Scanning {len(mu_vals)} × {len(R_vals)} parameter combinations...")
    
    for i, mu in enumerate(mu_vals):
        for j, R in enumerate(R_vals):
            E_avail = available_negative_energy(mu, tau, R)
            E_req = warp_energy_requirement(R, v)
            
            ratio = abs(E_avail) / E_req if E_req != 0 else 0
            feasibility_grid[i, j] = ratio
            
            # Track best configuration
            if ratio > best_ratio:
                best_ratio = ratio
                best_config = (mu, R, E_avail, E_req)
            
            # Track feasible configurations
            if ratio >= 1.0:
                feasible_configs.append((mu, R, ratio))
        
        # Progress indicator
        print(f"Progress: {100 * (i + 1) / len(mu_vals):.1f}%", end='\r')
    
    print()  # New line
    
    # Results
    print(f"\n📊 SCAN RESULTS:")
    print(f"Best feasibility ratio: {best_ratio:.3f}")
    if best_config:
        mu_best, R_best, E_av, E_req = best_config
        print(f"Best configuration: μ={mu_best:.3f}, R={R_best:.3f}")
        print(f"  Available energy: {E_av:.3e}")
        print(f"  Required energy: {E_req:.3e}")
    
    print(f"Feasible configurations (ratio ≥ 1.0): {len(feasible_configs)}")
    
    if feasible_configs:
        print("✅ FEASIBLE CONFIGURATIONS FOUND:")
        for mu, R, ratio in feasible_configs[:5]:  # Show first 5
            print(f"  μ={mu:.3f}, R={R:.3f}, ratio={ratio:.3f}")
        if len(feasible_configs) > 5:
            print(f"  ... and {len(feasible_configs) - 5} more")
    else:
        print("⚠️ No fully feasible configurations found")
        
        # Find configurations closest to feasibility
        max_ratio = np.max(feasibility_grid)
        mu_idx, R_idx = np.unravel_index(np.argmax(feasibility_grid), feasibility_grid.shape)
        mu_close = mu_vals[mu_idx]
        R_close = R_vals[R_idx]
        
        print(f"\nClosest to feasibility:")
        print(f"  μ={mu_close:.3f}, R={R_close:.3f}, ratio={max_ratio:.3f}")
        print(f"  Need {1/max_ratio:.1f}× more negative energy")
    
    # Visualization
    plt.figure(figsize=(12, 10))
    
    # 3D surface plot of feasibility ratio
    plt.subplot(2, 2, 1)
    MU, R_GRID = np.meshgrid(mu_vals, R_vals, indexing='ij')
    levels = np.linspace(0, np.max(feasibility_grid), 20)
    contour = plt.contourf(MU, R_GRID, feasibility_grid, levels=levels, cmap='RdYlGn')
    plt.colorbar(contour, label='Feasibility Ratio')
    plt.xlabel('μ (Polymer Parameter)')
    plt.ylabel('R (Bubble Radius)')
    plt.title('Feasibility Ratio E_avail/E_req')
    
    # Mark feasible region
    feasible_mask = feasibility_grid >= 1.0
    if np.any(feasible_mask):
        plt.contour(MU, R_GRID, feasibility_grid, levels=[1.0], colors='black', linewidths=2)
    
    # Mark best point
    if best_config:
        plt.plot(best_config[0], best_config[1], 'k*', markersize=15, label='Best')
        plt.legend()
    
    # Cross-section at best μ
    plt.subplot(2, 2, 2)
    if best_config:
        mu_best_idx = np.argmin(np.abs(mu_vals - best_config[0]))
        plt.plot(R_vals, feasibility_grid[mu_best_idx, :], 'b-', linewidth=2)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Feasible threshold')
        plt.xlabel('R (Bubble Radius)')
        plt.ylabel('Feasibility Ratio')
        plt.title(f'Cross-section at μ = {best_config[0]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Cross-section at best R
    plt.subplot(2, 2, 3)
    if best_config:
        R_best_idx = np.argmin(np.abs(R_vals - best_config[1]))
        plt.plot(mu_vals, feasibility_grid[:, R_best_idx], 'g-', linewidth=2)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Feasible threshold')
        plt.xlabel('μ (Polymer Parameter)')
        plt.ylabel('Feasibility Ratio')
        plt.title(f'Cross-section at R = {best_config[1]:.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    plt.hist(feasibility_grid.flatten(), bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='Feasibility threshold')
    plt.xlabel('Feasibility Ratio')
    plt.ylabel('Count')
    plt.title('Distribution of Feasibility Ratios')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_scan_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return feasibility_grid, mu_vals, R_vals, best_config

def main():
    """Main scanning routine."""
    print("🚀 AUTOMATED PARAMETER SCAN")
    print("Finding optimal μ and R for warp bubble feasibility")
    print("=" * 55)
    
    # Run parameter scan
    feasibility_grid, mu_vals, R_vals, best_config = parameter_scan()
    
    print(f"\n🎯 CONCLUSIONS:")
    if best_config and best_config[0] is not None:
        shortage_factor = 1.0 / (abs(best_config[2]) / best_config[3])
        print(f"Best configuration needs {shortage_factor:.1f}× more negative energy")
        
        if shortage_factor <= 2.0:
            print("✅ Close to feasibility! Consider:")
            print("   1. Cavity enhancement techniques")
            print("   2. Optimized field configurations")
            print("   3. Multi-cavity interference effects")
        else:
            print("⚠️ Significant gap remains. Next steps:")
            print("   1. Replace toy model with actual LQG solutions")
            print("   2. Include metric backreaction effects")
            print("   3. Explore squeezed vacuum enhancement")
    
    print(f"\n📈 Visualization saved as 'parameter_scan_results.png'")
    
    return feasibility_grid, best_config

if __name__ == "__main__":
    # Run the automated parameter scan
    grid, best = main()
    
    if best:
        mu_opt, R_opt = best[0], best[1]
        print(f"\n🔬 Recommended next experiment:")
        print(f"   Target μ ≈ {mu_opt:.3f}")
        print(f"   Design R ≈ {R_opt:.3f} Planck lengths")
        print("   Implement cavity-enhanced negative energy generation")
