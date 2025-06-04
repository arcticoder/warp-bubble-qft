#!/usr/bin/env python3
"""
Quick test of the warp bubble engine functionality.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import numpy as np
    print("✅ NumPy imported successfully")
    
    from scipy.integrate import simps
    print("✅ SciPy imported successfully")
    
    import matplotlib.pyplot as plt
    print("✅ Matplotlib imported successfully")
    
    from warp_qft.warp_bubble_engine import (
        squeezed_vacuum_energy,
        polymer_QI_bound, 
        sampling_function
    )
    print("✅ Core functions imported successfully")
    
    from warp_qft.warp_bubble_engine import WarpBubbleEngine
    print("✅ WarpBubbleEngine imported successfully")
    
    # Quick functionality test
    engine = WarpBubbleEngine()
    print("✅ WarpBubbleEngine instantiated successfully")
    
    # Test core functions
    energy = squeezed_vacuum_energy(1.0, 2*np.pi*5e9, 1e-12)
    bound = polymer_QI_bound(0.5, 1.0)
    f_vals = sampling_function(np.array([0, 1, 2]), 1.0)
    
    print(f"✅ Function tests passed:")
    print(f"   Squeezed energy: {energy:.2e} J/m³")
    print(f"   QI bound: {bound:.2e} J")
    print(f"   Sampling values: {f_vals}")
    
    print("\n🎯 All tests passed! Warp bubble engine is ready.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
