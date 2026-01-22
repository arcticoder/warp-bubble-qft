#!/usr/bin/env python3
"""
Quantum Inequality & Energy Density Verification Script

This script verifies the fundamental physics calculations underlying the
warp-bubble-qft framework, specifically:
1. Polymer energy density sign conventions
2. Ford-Roman quantum inequality (QI) violations
3. Physical consistency checks

Usage:
    python verify_qi_energy_density.py [--mu MU] [--save-plots]
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.warp_qft.lqg_profiles import (
    polymer_field_profile, lqg_negative_energy, toy_negative_energy
)


def verify_energy_density_sign(mu: float = 0.1, R: float = 2.3) -> dict:
    """
    Verify that polymer field energy density is properly negative.
    
    The README claims ρ_i < 0 via the formula:
        ρ_i = (1/2) [ sin²(μ̄ p_i)/μ̄² + (∇_d φ)_i² ]
    
    BUT: This is a sum of squares, inherently ≥ 0. The code must apply
    a negative sign externally or use a different convention.
    
    Returns:
        dict with energy values, spatial profile, and sign checks
    """
    x = np.linspace(-R, R, 500)
    rho_profile = polymer_field_profile(x, mu, R)
    
    # Integration over space
    dx = x[1] - x[0]
    total_energy = np.sum(rho_profile) * dx
    
    # Check sign consistency
    all_negative = np.all(rho_profile <= 0)
    peak_negative = np.min(rho_profile) < 0
    integral_negative = total_energy < 0
    
    return {
        "mu": mu,
        "R": R,
        "x_grid": x,
        "rho_profile": rho_profile,
        "total_energy": total_energy,
        "peak_density": np.min(rho_profile),
        "all_negative": all_negative,
        "peak_negative": peak_negative,
        "integral_negative": integral_negative,
        "sign_consistent": all_negative and integral_negative,
    }


def verify_ford_roman_qi(mu: float = 0.3, R: float = 2.3, Delta_t: float = 1.0) -> dict:
    """
    Verify Ford-Roman quantum inequality violation claim.
    
    Classical Ford-Roman QI for massless scalar:
        ∫ ρ(t) w(t) dt ≥ -C / (Δt)⁴
    where w(t) is a temporal sampling function (e.g., Lorentzian).
    
    The repo claims polymer fields violate this by producing
    I_polymer - I_classical < 0 over Δt beyond the Ford-Roman bound.
    
    Returns:
        dict with QI integral results and violation status
    """
    # Compute energies
    E_toy = toy_negative_energy(mu, R)
    E_lqg = lqg_negative_energy(mu, R, profile="polymer_field")
    
    # Ford-Roman bound (order-of-magnitude estimate for massless scalar)
    # Exact bound depends on sampling function details
    C_FR = 1.0 / (16 * np.pi**2)  # Typical coefficient
    FR_bound = -C_FR / (Delta_t**4)
    
    # Sampled integral approximation (simple box averaging over Δt)
    I_toy = E_toy * Delta_t
    I_lqg = E_lqg * Delta_t
    
    # Violation check: does LQG produce more negative integral than bound allows?
    violates_classical = I_lqg < I_toy
    violates_FR_bound = I_lqg < FR_bound
    
    return {
        "mu": mu,
        "R": R,
        "Delta_t": Delta_t,
        "E_toy": E_toy,
        "E_lqg": E_lqg,
        "I_toy": I_toy,
        "I_lqg": I_lqg,
        "Ford_Roman_bound": FR_bound,
        "difference": I_lqg - I_toy,
        "violates_classical": violates_classical,
        "violates_FR_bound": violates_FR_bound,
        "enhancement_ratio": abs(I_lqg / I_toy) if I_toy != 0 else 1.0,
    }


def scan_qi_over_parameters(
    mu_range: tuple = (0.05, 0.5),
    R_fixed: float = 2.3,
    N: int = 30,
) -> dict:
    """
    Scan QI violation status across polymer scale parameter μ.
    
    Returns:
        dict with scan results
    """
    mu_vals = np.linspace(mu_range[0], mu_range[1], N)
    results = []
    
    for mu in mu_vals:
        qi_result = verify_ford_roman_qi(mu, R_fixed)
        results.append({
            "mu": mu,
            "I_lqg": qi_result["I_lqg"],
            "I_toy": qi_result["I_toy"],
            "difference": qi_result["difference"],
            "violates": qi_result["violates_classical"],
        })
    
    return {
        "mu_range": mu_vals,
        "R": R_fixed,
        "results": results,
    }


def plot_energy_density_profiles(verify_result: dict, save_path: str = None):
    """Plot spatial energy density profile."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = verify_result["x_grid"]
    rho = verify_result["rho_profile"]
    
    ax.plot(x, rho, 'b-', linewidth=2, label='Polymer Field ρ(x)')
    ax.axhline(0, color='red', linestyle='--', linewidth=1, label='ρ = 0')
    ax.fill_between(x, rho, 0, where=(rho < 0), alpha=0.3, color='blue', label='Negative Energy Region')
    
    ax.set_xlabel('Position x (Planck units)', fontsize=12)
    ax.set_ylabel('Energy Density ρ(x)', fontsize=12)
    ax.set_title(f'Polymer Field Energy Density Profile (μ={verify_result["mu"]:.3f}, R={verify_result["R"]:.2f})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_qi_scan(scan_result: dict, save_path: str = None):
    """Plot QI violation across parameter space."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    mu_vals = scan_result["mu_range"]
    results = scan_result["results"]
    
    I_lqg = [r["I_lqg"] for r in results]
    I_toy = [r["I_toy"] for r in results]
    diff = [r["difference"] for r in results]
    
    # Plot 1: Integrated energies
    ax1.plot(mu_vals, I_lqg, 'b-', linewidth=2, label='I_LQG (polymer)')
    ax1.plot(mu_vals, I_toy, 'g--', linewidth=2, label='I_classical (toy)')
    ax1.axhline(0, color='red', linestyle=':', linewidth=1)
    ax1.set_xlabel('Polymer Scale μ', fontsize=12)
    ax1.set_ylabel('Integrated Energy I', fontsize=12)
    ax1.set_title('QI Integral vs Polymer Scale', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Difference (violation strength)
    ax2.plot(mu_vals, diff, 'r-', linewidth=2)
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, label='No violation')
    ax2.fill_between(mu_vals, diff, 0, where=(np.array(diff) < 0), alpha=0.3, color='red', label='QI Violation (I_LQG < I_toy)')
    ax2.set_xlabel('Polymer Scale μ', fontsize=12)
    ax2.set_ylabel('I_LQG - I_classical', fontsize=12)
    ax2.set_title('QI Violation Strength', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved QI scan plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Verify QI and energy density physics")
    parser.add_argument("--mu", type=float, default=0.1, help="Polymer scale parameter")
    parser.add_argument("--R", type=float, default=2.3, help="Bubble radius")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to results/")
    parser.add_argument("--scan", action="store_true", help="Run parameter scan over μ")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory for artifacts")
    
    args = parser.parse_args()
    
    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("QUANTUM INEQUALITY & ENERGY DENSITY VERIFICATION")
    print("=" * 60)
    
    # 1. Verify energy density signs
    print("\n1. Energy Density Sign Verification")
    print("-" * 60)
    energy_check = verify_energy_density_sign(args.mu, args.R)
    print(f"  Polymer scale μ: {energy_check['mu']:.3f}")
    print(f"  Bubble radius R: {energy_check['R']:.2f}")
    print(f"  Total integrated energy: {energy_check['total_energy']:.6f}")
    print(f"  Peak (most negative) density: {energy_check['peak_density']:.6f}")
    print(f"  All points negative: {energy_check['all_negative']}")
    print(f"  Integral negative: {energy_check['integral_negative']}")
    print(f"  Sign consistency: {'✅ PASS' if energy_check['sign_consistent'] else '❌ FAIL'}")
    
    # 2. Verify QI violation
    print("\n2. Ford-Roman Quantum Inequality Check")
    print("-" * 60)
    qi_check = verify_ford_roman_qi(args.mu, args.R)
    print(f"  Classical (toy) integral: {qi_check['I_toy']:.6f}")
    print(f"  LQG (polymer) integral: {qi_check['I_lqg']:.6f}")
    print(f"  Ford-Roman bound: {qi_check['Ford_Roman_bound']:.6e}")
    print(f"  Difference (I_LQG - I_toy): {qi_check['difference']:.6f}")
    print(f"  Violates classical: {qi_check['violates_classical']}")
    print(f"  Violates FR bound: {qi_check['violates_FR_bound']}")
    print(f"  Enhancement ratio: {qi_check['enhancement_ratio']:.2f}×")
    
    # 3. Optional: parameter scan
    if args.scan:
        print("\n3. QI Parameter Scan (μ ∈ [0.05, 0.5])")
        print("-" * 60)
        scan_result = scan_qi_over_parameters()
        num_violations = sum(1 for r in scan_result["results"] if r["violates"])
        print(f"  Scanned {len(scan_result['results'])} μ values")
        print(f"  QI violations found: {num_violations}/{len(scan_result['results'])}")
        
        if args.save_plots:
            plot_qi_scan(scan_result, str(results_dir / "qi_scan.png"))
    
    # 4. Save plots if requested
    if args.save_plots:
        plot_energy_density_profiles(energy_check, str(results_dir / "energy_density_profile.png"))
    
    print("\n" + "=" * 60)
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
