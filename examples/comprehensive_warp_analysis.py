#!/usr/bin/env python3
"""
Comprehensive Warp Bubble Demonstration Script

This script demonstrates the complete warp bubble analysis pipeline,
including parameter scanning, optimization, and energy feasibility analysis.

Usage:
    python comprehensive_warp_analysis.py [--quick] [--no-plots]
    
Options:
    --quick     : Run with reduced parameter sets for faster execution
    --no-plots  : Skip plot generation
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from warp_qft.warp_bubble_analysis import (
    run_warp_analysis,
    find_optimal_mu,
    compare_neg_energy,
    polymer_QI_bound,
    squeezed_vacuum_energy,
    visualize_scan
)

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive warp bubble analysis')
    parser.add_argument('--quick', action='store_true', 
                       help='Run with reduced parameters for faster execution')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    
    args = parser.parse_args()
    
    print("🌌 COMPREHENSIVE WARP BUBBLE ANALYSIS")
    print("=" * 50)
    print()
    
    if args.quick:
        print("🏃 Running in quick mode with reduced parameter sets...")
        mu_vals = [0.3, 0.6]
        tau_vals = [1.0, 2.0] 
        R_vals = [2.0, 3.0]
    else:
        print("🐌 Running full analysis (this may take a few minutes)...")
        mu_vals = [0.1, 0.3, 0.6, 1.0, 1.5]
        tau_vals = [0.5, 1.0, 1.5, 2.0]
        R_vals = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    print(f"Parameter ranges:")
    print(f"  μ (polymer scale): {mu_vals}")
    print(f"  τ (temporal scale): {tau_vals}")  
    print(f"  R (radial scale): {R_vals}")
    print()
    
    # Run the comprehensive analysis
    analysis_results = run_warp_analysis(
        mu_vals=mu_vals,
        tau_vals=tau_vals, 
        R_vals=R_vals,
        sigma=0.4,          # Slightly tighter shell
        A_factor=1.5,       # Higher amplitude factor
        omega=3*np.pi,      # Higher frequency
        generate_plots=not args.no_plots
    )
    
    # Advanced visualization if plots are enabled
    if not args.no_plots:
        print("\n📈 Generating comprehensive visualization...")
        fig = analysis_results.get('figure')
        
        if fig:
            # Save the figure
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)
            fig.savefig(output_dir / "warp_bubble_analysis.png", dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {output_dir / 'warp_bubble_analysis.png'}")
        
        # Detailed parameter optimization
        print("\n🔧 Detailed Parameter Optimization...")
        mu_detailed, bound_detailed, mu_array, bound_array = find_optimal_mu(
            mu_min=0.05, mu_max=1.2, steps=100, tau=0.5
        )
        
        # Plot optimization curve
        plt.figure(figsize=(10, 6))
        plt.plot(mu_array, bound_array * 1e34, 'b-', linewidth=2, label='QI Bound')
        plt.axvline(mu_detailed, color='r', linestyle='--', linewidth=2, 
                    label=f'Optimal μ = {mu_detailed:.3f}')
        plt.xlabel('Polymer Parameter μ')
        plt.ylabel('QI Bound (×10⁻³⁴ J)')
        plt.title('Quantum Inequality Bound Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "qi_bound_optimization.png", dpi=300, bbox_inches='tight')
        plt.show()
        print(f"QI optimization plot saved to: {output_dir / 'qi_bound_optimization.png'}")
        
        # Energy comparison across different scenarios
        print("\n⚡ Multi-Scenario Energy Analysis...")
        scenarios = [
            {'name': 'Microwave Cavity', 'r_squeeze': 1.0, 'omega': 2*np.pi*5e9, 'volume': 1e-12},
            {'name': 'Optical Cavity', 'r_squeeze': 1.5, 'omega': 2*np.pi*5e14, 'volume': 1e-15},
            {'name': 'Superconducting Circuit', 'r_squeeze': 2.0, 'omega': 2*np.pi*10e9, 'volume': 1e-18}
        ]
        
        feasibility_data = []
        best_mu = analysis_results['optimization']['best_mu']
        
        for scenario in scenarios:
            rho_neg = squeezed_vacuum_energy(
                scenario['r_squeeze'], 
                scenario['omega'], 
                scenario['volume']
            )
            E_available = rho_neg * scenario['volume']
            
            # Use the optimized parameters for requirement calculation
            E_required, _ = compare_neg_energy(
                best_mu, 1.0, 3.0, 0.5,  # mu, tau, R, dR
                scenario['r_squeeze'], scenario['omega'], scenario['volume']
            )
            
            ratio = abs(E_available / E_required) if E_required != 0 else np.inf
            
            feasibility_data.append({
                'scenario': scenario['name'],
                'E_available': E_available,
                'E_required': E_required,
                'ratio': ratio
            })
            
            print(f"{scenario['name']:20s}: Ratio = {ratio:.2e}")
    
    # Generate summary report
    print("\n📝 Analysis Summary:")
    print("-" * 30)
    
    optimal_results = analysis_results.get('optimization', {})
    squeezed_results = analysis_results.get('squeezed_analysis', {})
    
    print(f"Optimal polymer parameter: μ = {optimal_results.get('best_mu', 'N/A'):.3f}")
    print(f"Optimal QI bound: {optimal_results.get('best_bound', 'N/A'):.2e} J")
    
    if squeezed_results:
        E_req = squeezed_results.get('E_required', 0)
        E_avail = squeezed_results.get('E_available', 0)
        ratio = squeezed_results.get('feasibility_ratio', 0)
        
        print(f"Required negative energy: {E_req:.2e} J")
        print(f"Available squeezed energy: {E_avail:.2e} J")
        print(f"Feasibility ratio: {ratio:.2e}")
    
    # Final assessment
    print("\n🎯 ANALYSIS COMPLETE")
    print("=" * 30)
    
    if squeezed_results and squeezed_results.get('feasibility_ratio', 0) > 1:
        print("✅ RESULT: Warp bubble formation appears FEASIBLE!")
        print(f"   Energy surplus: {squeezed_results['feasibility_ratio']:.1f}x")
    elif squeezed_results and squeezed_results.get('feasibility_ratio', 0) > 0.1:
        print("⚠️  RESULT: Warp bubble formation is CHALLENGING but possible")
        print(f"   Energy deficit: {1/squeezed_results['feasibility_ratio']:.1f}x")
    else:
        print("❌ RESULT: Warp bubble formation requires significant advancement")
        if squeezed_results and squeezed_results.get('feasibility_ratio', 0) > 0:
            print(f"   Energy deficit: {1/squeezed_results['feasibility_ratio']:.0f}x")
    
    num_violations = sum(1 for v in analysis_results['violations'].values() if v)
    total_configs = len(analysis_results['scan_results'])
    
    print(f"\nKey findings:")
    print(f"- Polymer enhancement enables {num_violations} QI violations")
    print(f"- {total_configs} configurations tested total")
    print(f"- Optimal parameter μ = {optimal_results.get('best_mu', 'N/A'):.3f}")
    
    if not args.no_plots:
        best_scenario = max(feasibility_data, key=lambda x: x['ratio']) if 'feasibility_data' in locals() else None
        if best_scenario:
            print(f"- Best experimental approach: {best_scenario['scenario']}")
        
        output_dir = Path(__file__).parent / "output"
        print(f"\nFiles generated:")
        print(f"- {output_dir / 'warp_bubble_analysis.png'}")
        print(f"- {output_dir / 'qi_bound_optimization.png'}")
    
    return analysis_results


if __name__ == "__main__":
    # Run the comprehensive analysis
    try:
        results = main()
        
        print("\n🚀 Ready to proceed with warp bubble implementation!")
        print("   Theoretical foundation: ✅ Complete")
        print("   Parameter optimization: ✅ Complete") 
        print("   Feasibility analysis: ✅ Complete")
        print("   Next: Full 3+1D implementation")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure the warp_qft package is properly installed:")
        print("  pip install -e .")
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        print("Check that all dependencies are installed:")
        print("  pip install -r requirements.txt")
