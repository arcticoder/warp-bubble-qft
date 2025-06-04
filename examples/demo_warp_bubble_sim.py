#!/usr/bin/env python3
"""
Demo: Warp Bubble Negative-Energy Simulation

This script demonstrates the formation and evolution of stable negative energy 
densities (warp bubbles) in a polymer-quantized field theory.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from warp_qft.field_algebra import PolymerField, analyze_negative_energy_formation
from warp_qft.negative_energy import compute_negative_energy_region, WarpBubble
from warp_qft.stability import ford_roman_bounds, polymer_modified_bounds, violation_duration

def main():
    """Run the warp bubble simulation demo."""
    print("🌌 Warp Bubble QFT Demonstration")
    print("=" * 50)
    
    # Simulation parameters
    lattice_size = 64
    polymer_scales = [0.0, 0.3, 0.6, 1.0]  # Compare different polymer regimes
    field_amplitude = 1.5
    
    results = {}
    
    # Run simulations for different polymer scales
    for mu_bar in polymer_scales:
        print(f"\n🔬 Analyzing polymer scale μ̄ = {mu_bar}")
        
        # Compute negative energy regions
        result = compute_negative_energy_region(lattice_size, mu_bar, field_amplitude)
        
        if result["total_negative_energy"] < 0:
            print(f"✅ Negative energy found: {result['total_negative_energy']:.6f}")
            print(f"   Negative sites: {result['negative_sites']}")
            
            if result["bubble"]:
                bubble = result["bubble"]
                print(f"   Bubble center: {bubble.center:.3f}")
                print(f"   Bubble radius: {bubble.radius:.3f}")
                print(f"   Peak density: {bubble.rho_neg:.6f}")
                
                # Stability analysis
                stability = result["stability_analysis"]
                if stability:
                    print(f"   Classical lifetime: {stability['classical_lifetime']:.6f}")
                    print(f"   Polymer lifetime: {stability['polymer_lifetime']:.6f}")
                    print(f"   Enhancement factor: {stability['stabilization_factor']:.3f}")
        else:
            print("❌ No negative energy regions found")
        
        results[mu_bar] = result
    
    # Visualization
    create_visualization(results, polymer_scales)
    
    # Ford-Roman bound analysis
    print("\n📊 Ford-Roman Bound Analysis")
    print("-" * 30)
    
    spatial_scale = 0.1
    test_energy = -0.5
    
    for mu_bar in [0.0, 0.3, 0.6]:
        classical_bounds = ford_roman_bounds(test_energy, spatial_scale)
        polymer_bounds = polymer_modified_bounds(test_energy, spatial_scale, mu_bar)
        
        print(f"\nμ̄ = {mu_bar}:")
        print(f"  Classical bound: {classical_bounds['ford_roman_bound']:.6f}")
        print(f"  Polymer bound: {polymer_bounds['ford_roman_bound']:.6f}")
        print(f"  Enhancement: {polymer_bounds['enhancement_factor']:.3f}×")
        
        duration = violation_duration(test_energy, spatial_scale, mu_bar)
        print(f"  Max duration: {duration['max_duration']:.6f}")
    
    print("\n🎯 Demonstration completed!")
    print("Check the generated plots for visualization of results.")


def create_visualization(results, polymer_scales):
    """Create visualization of the simulation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Warp Bubble Formation in Polymer QFT", fontsize=16)
    
    # Plot 1: Energy density profiles
    ax1 = axes[0, 0]
    for mu_bar in polymer_scales:
        if mu_bar in results and results[mu_bar]["total_negative_energy"] < 0:
            x_grid = results[mu_bar]["x_grid"]
            energy = results[mu_bar]["energy_density"]
            ax1.plot(x_grid, energy, label=f"μ̄ = {mu_bar}", linewidth=2)
    
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Position")
    ax1.set_ylabel("Energy Density")
    ax1.set_title("Energy Density Profiles")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total negative energy vs polymer scale
    ax2 = axes[0, 1]
    neg_energies = []
    mu_values = []
    
    for mu_bar in polymer_scales:
        if mu_bar in results:
            neg_energy = results[mu_bar]["total_negative_energy"]
            if neg_energy < 0:
                neg_energies.append(abs(neg_energy))
                mu_values.append(mu_bar)
    
    if neg_energies:
        ax2.plot(mu_values, neg_energies, 'bo-', linewidth=2, markersize=8)
        ax2.set_xlabel("Polymer Scale μ̄")
        ax2.set_ylabel("|Total Negative Energy|")
        ax2.set_title("Negative Energy vs Polymer Scale")
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Enhancement factors
    ax3 = axes[1, 0]
    enhancement_factors = []
    mu_test = np.linspace(0.1, 1.0, 20)
    
    for mu in mu_test:
        bounds = polymer_modified_bounds(-0.5, 0.1, mu)
        enhancement_factors.append(bounds["enhancement_factor"])
    
    ax3.plot(mu_test, enhancement_factors, 'r-', linewidth=2)
    ax3.set_xlabel("Polymer Scale μ̄")
    ax3.set_ylabel("Enhancement Factor")
    ax3.set_title("Ford-Roman Bound Enhancement")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Stability duration
    ax4 = axes[1, 1]
    durations = []
    
    for mu in mu_test:
        duration_result = violation_duration(-0.5, 0.1, mu)
        durations.append(duration_result["max_duration"])
    
    ax4.plot(mu_test, durations, 'g-', linewidth=2)
    ax4.set_xlabel("Polymer Scale μ̄")
    ax4.set_ylabel("Maximum Duration")
    ax4.set_title("Violation Duration vs Polymer Scale")
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("warp_bubble_demo.png", dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
