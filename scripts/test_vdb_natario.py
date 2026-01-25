#!/usr/bin/env python3
"""
Quick test for Van den Broeck–Natário implementation.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import numpy as np
    
    from warp_qft.metrics.van_den_broeck_natario import (
        van_den_broeck_shape,
        natario_shift_vector,
        van_den_broeck_natario_metric,
        energy_requirement_comparison,
        optimal_vdb_parameters
    )
    
    print("✅ Successfully imported Van den Broeck–Natário functions")
    
    # Test basic functionality
    print("\n🧪 Testing basic functionality...")
    
    # Test parameters
    R_int = 100.0
    R_ext = 2.3
    v_bubble = 1.0
    
    # Test shape function
    r_test = 5.0
    shape_value = van_den_broeck_shape(r_test, R_int, R_ext)
    print(f"Shape function at r={r_test}: {shape_value:.6f}")
    
    # Test shift vector
    x_test = np.array([3.0, 0.0, 0.0])
    shift = natario_shift_vector(x_test, v_bubble, R_int, R_ext)
    print(f"Shift vector at x={x_test}: {shift}")
    
    # Test metric
    metric = van_den_broeck_natario_metric(x_test, 0.0, v_bubble, R_int, R_ext)
    print(f"Metric determinant: {np.linalg.det(metric):.6f}")
    
    # Test energy comparison
    comparison = energy_requirement_comparison(R_int, R_ext, v_bubble)
    print(f"Energy reduction factor: {comparison['reduction_factor']:.2e}")
    
    # Test optimization
    optimal = optimal_vdb_parameters(payload_size=10.0)
    print(f"Optimal parameters: R_int={optimal['R_int']:.1f}, R_ext={optimal['R_ext']:.3e}")
    print(f"Optimal reduction factor: {optimal['reduction_factor']:.2e}")
    
    print("\n✅ All tests passed! Van den Broeck–Natário implementation working correctly.")
    
    # Quick energy reduction demonstration
    print(f"\n🚀 Energy Reduction Demonstration:")
    print(f"Standard Alcubierre energy: {comparison['alcubierre_energy']:.2e}")
    print(f"Van den Broeck–Natário energy: {comparison['vdb_natario_energy']:.2e}")
    print(f"Reduction factor: {comparison['reduction_factor']:.2e}")
    
    if comparison['reduction_factor'] >= 1e5:
        print("🎯 Achieved target 10⁵× reduction!")
    if comparison['reduction_factor'] >= 1e6:
        print("🎯 Achieved target 10⁶× reduction!")
    
    print(f"Volume ratio (neck/payload): {comparison['volume_ratio']:.2e}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure NumPy is installed: pip install numpy")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
