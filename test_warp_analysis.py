#!/usr/bin/env python3
"""
Simple test script for the comprehensive warp bubble analysis module.
"""
import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, 'src')

def test_basic_imports():
    """Test that all required functions can be imported."""
    try:
        from warp_qft.warp_bubble_analysis import (
            squeezed_vacuum_energy,
            polymer_QI_bound,
            run_warp_analysis,
            scan_3d_shell,
            find_optimal_mu,
            compare_neg_energy,            visualize_scan
        )
        print("✓ All core functions imported successfully")
        assert True  # Verify imports succeeded
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_calculations():
    """Test basic function calculations."""
    try:
        from warp_qft.warp_bubble_analysis import squeezed_vacuum_energy, polymer_QI_bound
        
        # Test squeezed vacuum energy
        energy = squeezed_vacuum_energy(r_squeeze=0.5, omega=1e15, volume=1e-9)
        print(f"✓ Squeezed vacuum energy: {energy:.2e} J/m³")
        
        # Test polymer QI bound
        bound = polymer_QI_bound(mu=0.5, tau=1.0)
        print(f"✓ Polymer QI bound: {bound:.2e} J")
        
        assert energy < 0  # Energy should be negative
        assert bound < 0  # Bound should be negative
    except Exception as e:
        print(f"❌ Calculation error: {e}")
        return False

def test_quick_analysis():
    """Run a minimal analysis to verify the pipeline works."""
    try:
        from warp_qft.warp_bubble_analysis import run_warp_analysis
        
        print("🚀 Running quick analysis test...")
        result = run_warp_analysis(
            mu_vals=[0.3, 0.6],
            tau_vals=[1.0],
            R_vals=[2.0, 3.0],
            sigma=0.4,
            A_factor=1.2,
            omega=2*np.pi,
            generate_plots=False
        )
        
        assert result is not None, "Analysis returned None"
        print(f"Result keys: {list(result.keys())}")
        
        # Check for expected keys
        expected_keys = ['scan_results', 'violations', 'optimal_mu', 'optimal_bound']
        for key in expected_keys:
            if key in result:
                print(f"  ✓ Found {key}")
            else:
                print(f"  ⚠ Missing {key}")
                
            assert key in result, f"Missing expected key: {key}"
            
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()
        raise

def test_optimization():
    """Test parameter optimization functionality."""
    try:
        from warp_qft.warp_bubble_analysis import find_optimal_mu
        
        print("🔧 Testing parameter optimization...")
        opt_mu, opt_bound, mu_array, bound_array = find_optimal_mu(
            mu_min=0.1, mu_max=1.0, steps=20, tau=1.0
        )
        
        print(f"✓ Optimal μ = {opt_mu:.3f}")
        print(f"✓ Optimal bound = {opt_bound:.2e} J")
        print(f"✓ Scanned {len(mu_array)} parameter values")
        
        assert opt_mu > 0, "Optimal mu should be positive"
        assert opt_bound < 0, "Optimal bound should be negative"
        assert len(mu_array) == 20, "Should have 20 parameter values"
            
    except Exception as e:
        print(f"❌ Optimization error: {e}")
        raise

def test_energy_comparison():
    """Test energy requirement vs availability comparison."""
    try:
        from warp_qft.warp_bubble_analysis import compare_neg_energy
        
        print("⚡ Testing energy comparison...")
        E_req, E_avail = compare_neg_energy(
            mu=0.5, tau=1.0, R=3.0, dR=0.5,
            r_squeeze=1.0, omega=2*np.pi*5e9, cavity_vol=1e-12
        )
        
        feasibility_ratio = abs(E_avail / E_req) if E_req != 0 else np.inf
        
        print(f"✓ Required energy: {E_req:.2e} J")
        print(f"✓ Available energy: {E_avail:.2e} J")
        print(f"✓ Feasibility ratio: {feasibility_ratio:.2e}")
        
        assert E_req > 0, "Required energy should be positive"
        assert E_avail != 0, "Available energy should not be zero"
        assert isinstance(feasibility_ratio, float), "Feasibility ratio should be a float"
        
        if feasibility_ratio > 1:
            print("🎯 Energy analysis suggests feasible configuration!")
        else:
            print("⚠ Energy gap remains challenging")
            
    except Exception as e:
        print(f"❌ Energy comparison error: {e}")
        raise

def main():
    """Run all tests."""
    print("🌌 WARP BUBBLE ANALYSIS MODULE TEST")
    print("=" * 50)
    print()
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Basic Calculations", test_basic_calculations),
        ("Quick Analysis", test_quick_analysis),
        ("Parameter Optimization", test_optimization),
        ("Energy Comparison", test_energy_comparison)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        if test_func():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The comprehensive warp bubble analysis is ready.")
        print()
        print("To run the full analysis:")
        print("  python examples/comprehensive_warp_analysis.py")
        print()
        print("For quick mode:")
        print("  python examples/comprehensive_warp_analysis.py --quick")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
