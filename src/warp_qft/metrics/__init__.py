"""
Warp Bubble Metrics Package

This package implements various warp bubble metric formulations:
- Van den Broeck–Natário hybrid metrics for volume reduction
- Standard Alcubierre metrics for comparison
- Energy tensor calculations and optimizations
"""

from .van_den_broeck_natario import (
    van_den_broeck_shape,
    natario_shift_vector,
    van_den_broeck_natario_metric,
    compute_energy_tensor,
    energy_requirement_comparison
)

__all__ = [
    'van_den_broeck_shape',
    'natario_shift_vector', 
    'van_den_broeck_natario_metric',
    'compute_energy_tensor',
    'energy_requirement_comparison'
]
