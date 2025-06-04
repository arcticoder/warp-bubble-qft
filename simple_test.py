#!/usr/bin/env python3
import sys
import os

# Add src directory to path
sys.path.insert(0, 'src')

print("Testing imports...")

try:
    import numpy as np
    print("✓ NumPy imported")    try:
        from scipy.integrate import simpson as simps
    except ImportError:
        from scipy.integrate import simps
    print("✓ SciPy imported")
    
    import matplotlib.pyplot as plt
    print("✓ Matplotlib imported")
    
    from warp_qft.warp_bubble_analysis import squeezed_vacuum_energy
    print("✓ Basic function imported")
    
    # Test calculation
    result = squeezed_vacuum_energy(0.5, 1e15, 1e-9)
    print(f"✓ Calculation successful: {result:.2e}")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
