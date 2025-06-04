#!/usr/bin/env python3
"""
Warp Bubble Power Analysis Demo

This script demonstrates the complete warp bubble power analysis framework,
including:

1. Toy negative energy density modeling
2. Available vs required energy comparison
3. Parameter optimization for feasibility
4. Comprehensive visualization

Based on the theoretical framework outlined in the user's request.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from warp_qft.warp_bubble_engine import (
    WarpBubbleEngine,
    toy_negative_energy_density,
    available_negative_energy,
    warp_energy_requirement,
    compute_feasibility_ratio,
    parameter_scan_feasibility,
    visualize_feasibility_scan,
    print_feasibility_summary,
    sampling_function
)

def demonstrate_toy_model():
    """Demonstrate the toy negative energy density model."""
    print("\n🧪 DEMONSTRATING TOY NEGATIVE ENERGY MODEL")
    print("=" * 50)
    
    # Parameters
    mu_vals = [0.1, 0.3, 0.6, 1.0]
    R = 2.0
    x = np.linspace(-R, R, 200)
    
    plt.figure(figsize=(12, 8))
    
    # Panel 1: Energy density profiles
    plt.subplot(2, 2, 1)
    for mu in mu_vals:
        rho = toy_negative_energy_density(x, mu, R)
        plt.plot(x, rho, label=f'μ = {mu:.1f}')
    plt.xlabel('x (Planck lengths)')
    plt.ylabel('ρ(x)')
    plt.title('Negative Energy Density Profiles')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Panel 2: Sampling function
    plt.subplot(2, 2, 2)
    tau_vals = [0.5, 1.0, 2.0]
    t = np.linspace(-5, 5, 200)
    for tau in tau_vals:
        f = sampling_function(t, tau)
        plt.plot(t, f, label=f'τ = {tau:.1f}')
    plt.xlabel('t')
    plt.ylabel('f(t,τ)')
    plt.title('Gaussian Sampling Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Panel 3: Available energy vs μ
    plt.subplot(2, 2, 3)
    mu_range = np.linspace(0.1, 1.5, 50)
    tau = 1.0
    E_avail = [available_negative_energy(mu, tau, R) for mu in mu_range]
    plt.plot(mu_range, E_avail, 'b-', linewidth=2)
    plt.xlabel('μ (Polymer Parameter)')
    plt.ylabel('Available Negative Energy')
    plt.title('Available Energy vs Polymer Parameter')
    plt.grid(True, alpha=0.3)
    
    # Panel 4: Required energy vs R
    plt.subplot(2, 2, 4)
    R_range = np.linspace(0.5, 5.0, 50)
    v = 1.0
    E_req = [warp_energy_requirement(R, v) for R in R_range]
    plt.plot(R_range, E_req, 'r-', linewidth=2)
    plt.xlabel('R (Bubble Radius)')
    plt.ylabel('Required Energy')
    plt.title('Required Energy vs Bubble Radius')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('warp_bubble_toy_model.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Toy model demonstration complete!")

def demonstrate_feasibility_analysis():
    """Demonstrate the complete feasibility analysis."""
    print("\n🎯 DEMONSTRATING FEASIBILITY ANALYSIS")
    print("=" * 50)
    
    # Example parameters from the user's request
    mu = 0.3    # polymer scale in optimal range
    tau = 1.0   # sampling width
    R = 1.0     # bubble radius (in Planck units)
    v = 1.0     # normalized warp-1 velocity
    
    print("\nExample calculation (matching user's request):")
    print(f"μ = {mu}, τ = {tau}, R = {R}, v = {v}")
    
    E_avail = available_negative_energy(mu, tau, R, Nx=500, Nt=500)
    E_req = warp_energy_requirement(R, v)
    
    print(f"Available Negative Energy: {E_avail:.3e}")
    print(f"Required Energy for R={R}, v={v}: {E_req:.3e}")
    print(f"Feasibility Ratio: {E_avail/E_req:.3e}")
    
    if abs(E_avail) >= E_req:
        print("✅ Configuration is feasible!")
    else:
        shortage = E_req / abs(E_avail)
        print(f"⚠️  Need {shortage:.1f}x more negative energy")

def run_parameter_optimization():
    """Run parameter optimization to find best configuration."""
    print("\n🔧 RUNNING PARAMETER OPTIMIZATION")
    print("=" * 50)
    
    # Scan parameters
    scan_results = parameter_scan_feasibility(
        mu_range=(0.1, 1.0),
        R_range=(0.5, 3.0),
        num_points=25,
        tau=1.0,
        v=1.0
    )
    
    # Print summary
    print_feasibility_summary(scan_results)
    
    # Generate visualization
    fig = visualize_feasibility_scan(scan_results)
    plt.savefig('warp_bubble_feasibility_scan.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return scan_results

def main():
    """Main demonstration routine."""
    print("🚀 WARP BUBBLE POWER ANALYSIS DEMONSTRATION")
    print("=" * 60)
    print("Quantifying negative energy requirements vs. availability")
    print("for macroscopic warp bubble formation using polymer QFT")
    print("=" * 60)
    
    try:
        # 1. Demonstrate toy model
        demonstrate_toy_model()
        
        # 2. Show feasibility calculation
        demonstrate_feasibility_analysis()
        
        # 3. Run optimization
        scan_results = run_parameter_optimization()
        
        # 4. Demonstrate engine functionality
        print("\n🏭 DEMONSTRATING WARP BUBBLE ENGINE")
        print("=" * 50)
        
        engine = WarpBubbleEngine()
        
        # Run power analysis
        power_results = engine.run_power_analysis(
            mu_range=(0.1, 0.8),
            R_range=(0.5, 4.0),
            num_points=15,
            tau=1.0,
            v=1.0,
            visualize=True
        )
        
        # Analyze specific promising configuration
        if power_results['best_params']:
            mu_best, R_best = power_results['best_params']
            detailed_results = engine.analyze_specific_configuration(
                mu=mu_best, tau=1.0, R=R_best, v=1.0, verbose=True
            )
            
            # Plot energy profile
            plt.figure(figsize=(10, 6))
            x = detailed_results['spatial_grid']
            rho = detailed_results['energy_profile']
            plt.plot(x, rho, 'b-', linewidth=2, label='Negative Energy Profile')
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel('x (Planck lengths)')
            plt.ylabel('ρ(x)')
            plt.title(f'Optimal Energy Profile (μ={mu_best:.3f}, R={R_best:.3f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('optimal_energy_profile.png', dpi=150, bbox_inches='tight')
            plt.show()
        
        print("\n🎯 ANALYSIS COMPLETE!")
        print("Generated visualizations:")
        print("  - warp_bubble_toy_model.png")
        print("  - warp_bubble_feasibility_scan.png")
        print("  - optimal_energy_profile.png")
        
        print("\n🔬 Key findings:")
        if power_results['best_ratio'] >= 1.0:
            print("  ✅ Warp bubble formation appears feasible!")
            print(f"  ✅ Best feasibility ratio: {power_results['best_ratio']:.3f}")
        else:
            print("  ⚠️  Additional negative energy sources needed")
            print(f"  ⚠️  Current best ratio: {power_results['best_ratio']:.3f}")
        
        print("\n📋 Next steps:")
        print("  1. Replace toy model with full LQG field solutions")
        print("  2. Implement Einstein field solver for accurate E_req")
        print("  3. Optimize experimental parameters")
        print("  4. Design cavity enhancement schemes")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
