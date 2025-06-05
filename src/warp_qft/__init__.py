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
from .warp_bubble_analysis import (
    run_warp_analysis,
    squeezed_vacuum_energy,
    scan_3d_shell,
    find_optimal_mu,
    compare_neg_energy,
    polymer_QI_bound,
    compute_I_3d,
    visualize_scan,
    evolve_phi_pi_3plus1D,
    linearized_stability,
    solve_warp_metric_3plus1D,
    generate_analysis_report
)
from .enhancement_pipeline import WarpBubbleEnhancementPipeline
from .enhancement_pathway import (
    EnhancementConfig,
    CavityBoostCalculator,
    QuantumSqueezingEnhancer,
    MultiBubbleSuperposition,
    ComprehensiveEnhancementCalculator
)
from .metrics import (
    van_den_broeck_shape,
    natario_shift_vector,
    van_den_broeck_natario_metric,
    compute_energy_tensor,
    energy_requirement_comparison
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
    "run_warp_analysis",
    "squeezed_vacuum_energy",
    "scan_3d_shell",
    "find_optimal_mu",
    "compare_neg_energy",
    "polymer_QI_bound",
    "compute_I_3d",
    "visualize_scan",
    "evolve_phi_pi_3plus1D",
    "linearized_stability",
    "solve_warp_metric_3plus1D",
    "generate_analysis_report",
    "WarpBubbleEnhancementPipeline",
    "EnhancementConfig",
    "CavityBoostCalculator",
    "QuantumSqueezingEnhancer",
    "MultiBubbleSuperposition",    "ComprehensiveEnhancementCalculator",
    "van_den_broeck_shape",
    "natario_shift_vector", 
    "van_den_broeck_natario_metric",
    "compute_energy_tensor",
    "energy_requirement_comparison"
]
