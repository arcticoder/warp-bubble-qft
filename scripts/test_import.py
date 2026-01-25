#!/usr/bin/env python3
"""
Quick test script to verify the warp bubble analysis module imports and runs correctly.
"""
import sys
import os
sys.path.insert(0, 'src')

try:
    from warp_qft.warp_bubble_analysis import (
        run_warp_analysis,
        squeezed_vacuum_energy,
        polymer_QI_bound
    )
    print("✓ Successfully imported warp bubble analysis functions")
    
    # Test a simple calculation
    energy = squeezed_vacuum_energy(r_squeeze=0.5, omega=1e15, volume=1e-9)
    print(f"✓ Squeezed vacuum energy calculation: {energy:.2e} J/m³")
    
    # Test QI bound calculation
    bound = polymer_QI_bound(mu=0.5, tau=1.0)
    print(f"✓ Polymer QI bound calculation: {bound:.2e} J")
    
    print("\n🚀 Running quick analysis test...")
    result = run_warp_analysis(
        mu_vals=[0.3, 0.6],
        tau_vals=[1.0],
        R_vals=[2.0],
        generate_plots=False
    )
    
    if result:
        print("✓ Analysis completed successfully!")
        print(f"Result keys: {list(result.keys())}")
        if 'violations' in result:
            print(f"Violations found: {len(result['violations'])}")
    else:
        print("⚠ Analysis returned None")

except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error during execution: {e}")
    import traceback
    traceback.print_exc()
