"""
Warp Bubble QFT Package

A quantum field theory implementation on polymer/loop backgrounds for stable 
negative-energy densities and warp bubble formation.
"""

__version__ = "0.1.0"
__author__ = "Arcticoder"

from .field_algebra import PolymerField, compute_commutator
from .negative_energy import WarpBubble, compute_negative_energy_region
from .stability import ford_roman_bounds, polymer_modified_bounds, violation_duration
from .bubble_stability import (
    compute_bubble_lifetime, 
    check_bubble_stability_conditions, 
    analyze_bubble_stability_theorem, 
    optimize_polymer_parameters
)
from .warp_bubble_engine import (
    WarpBubbleEngine,
    squeezed_vacuum_energy,
    polymer_QI_bound,
    find_optimal_mu,
    visualize_scan,
    run_warp_bubble_analysis
)

__all__ = [
    "PolymerField",
    "WarpBubble", 
    "compute_commutator",
    "compute_negative_energy_region",
    "ford_roman_bounds",
    "polymer_modified_bounds",
    "violation_duration",
    "compute_bubble_lifetime",
    "check_bubble_stability_conditions",
    "analyze_bubble_stability_theorem",
    "optimize_polymer_parameters",
    "WarpBubbleEngine",
    "squeezed_vacuum_energy", 
    "polymer_QI_bound",
    "find_optimal_mu",
    "visualize_scan",
    "run_warp_bubble_analysis"
]
