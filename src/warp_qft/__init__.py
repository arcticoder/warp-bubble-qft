"""
Warp Bubble QFT Package

A quantum field theory implementation on polymer/loop backgrounds for stable 
negative-energy densities and warp bubble formation.
"""

__version__ = "0.1.0"
__author__ = "Arcticoder"

from .field_algebra import PolymerField, compute_commutator
from .negative_energy import WarpBubble, compute_negative_energy_region
from .stability import ford_roman_bounds, violation_duration

__all__ = [
    "PolymerField",
    "WarpBubble", 
    "compute_commutator",
    "compute_negative_energy_region",
    "ford_roman_bounds",
    "violation_duration"
]
