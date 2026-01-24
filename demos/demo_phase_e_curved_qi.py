#!/usr/bin/env python3
"""
Demo: Phase E Curved-Space QI Refinements

Demonstrates three extensions to curved_qi_verification.py:
1. 4D proxy integral mode (extend 1+1D to 3+1D spacetime volume)
2. Normalized margin metric Δ̄ = (I-B)/|B|
3. Parameterized bound family (flat Ford-Roman, curved toy, hybrid)

Usage:
    python demos/demo_phase_e_curved_qi.py

Output:
    Comparison table showing all bound models and dimension modes
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from curved_qi_verification import verify_curved_qi


def run_phase_e_demo() -> None:
    """
    Run all Phase E configurations and display results.
    """
    print("\n" + "=" * 80)
    print("  Phase E Demo: Curved-Space QI Refinements")
    print("=" * 80)
    print("\nTesting three bound models × two dimension modes = 6 configurations")
    print("\nBound models:")
    print("  1. flat-ford-roman: -C/(Δt)^d (standard flat-space bound)")
    print("  2. curved-toy:      -C/R²     (heuristic curvature-dependent)")
    print("  3. hybrid:          max(flat, curved) (most restrictive)")
    print("\nDimension modes:")
    print("  - 1+1D: Standard integration (no transverse volume)")
    print("  - 4D proxy: Include spherical transverse volume factor 4πR²")
    print("\n" + "-" * 80)
    
    # Standard parameters
    mu = 0.3
    R_bubble = 2.3
    Delta_t = 1.0
    
    configurations = [
        # 1+1D modes
        {"use_4d_proxy": False, "bound_type": "flat-ford-roman", "label": "1+1D Flat"},
        {"use_4d_proxy": False, "bound_type": "curved-toy", "label": "1+1D Curved"},
        {"use_4d_proxy": False, "bound_type": "hybrid", "label": "1+1D Hybrid"},
        # 4D proxy modes
        {"use_4d_proxy": True, "bound_type": "flat-ford-roman", "label": "4D Flat"},
        {"use_4d_proxy": True, "bound_type": "curved-toy", "label": "4D Curved"},
        {"use_4d_proxy": True, "bound_type": "hybrid", "label": "4D Hybrid"},
    ]
    
    results = []
    
    print("\nRunning verifications...")
    for config in configurations:
        result = verify_curved_qi(
            mu=mu,
            R_bubble=R_bubble,
            sampling_width=Delta_t,
            metric_file=None,
            use_4d_proxy=config["use_4d_proxy"],
            bound_type=config["bound_type"]
        )
        result["config_label"] = config["label"]
        results.append(result)
        print(f"  ✓ {config['label']:12s} complete")
    
    # Display results table
    print("\n" + "-" * 80)
    print("Results Summary")
    print("-" * 80)
    print(f"{'Config':<12s} {'I_curved':>12s} {'Bound':>12s} {'Δ̄':>10s} {'Status':>12s}")
    print("-" * 80)
    
    for res in results:
        label = res["config_label"]
        I = res["integrals"]["curved_space"]
        B = res["bounds"]["curved_bound"]
        Delta_bar = res["violations"]["normalized_margin_curved"]
        violates = res["violations"]["violates_curved_bound"]
        status = "VIOLATES" if violates else "OK"
        
        print(f"{label:<12s} {I:+12.3e} {B:+12.3e} {Delta_bar:+10.3f} {status:>12s}")
    
    print("-" * 80)
    
    # Interpretation
    print("\nInterpretation:")
    print("\n1. Dimension Effect (4D vs 1+1D):")
    flat_1d = next(r for r in results if r["config_label"] == "1+1D Flat")
    flat_4d = next(r for r in results if r["config_label"] == "4D Flat")
    enhancement_1d = flat_1d["integrals"]["metric_enhancement_factor"]
    enhancement_4d = flat_4d["integrals"]["metric_enhancement_factor"]
    print(f"   - 1+1D metric enhancement: {enhancement_1d:.2f}×")
    print(f"   - 4D proxy enhancement:    {enhancement_4d:.2f}×")
    print(f"   - Volume factor increases integral by {enhancement_4d / enhancement_1d:.1f}×")
    
    print("\n2. Bound Model Effect:")
    curved_1d = next(r for r in results if r["config_label"] == "1+1D Curved")
    hybrid_1d = next(r for r in results if r["config_label"] == "1+1D Hybrid")
    print(f"   - Curved-toy bound (1+1D):  Δ̄ = {curved_1d['violations']['normalized_margin_curved']:+.3f} (no viol.)")
    print(f"   - Hybrid bound (1+1D):      Δ̄ = {hybrid_1d['violations']['normalized_margin_curved']:+.3f} (violation)")
    print("   - Hybrid picks restrictive flat bound → reveals violation")
    
    print("\n3. Normalized Margin Δ̄ = (I-B)/|B|:")
    print("   - Positive: integral above bound (no violation)")
    print("   - Negative: integral below bound (violation)")
    print("   - Magnitude: severity of violation (|Δ̄| >> 1 means strong violation)")
    
    print("\n4. Physical Interpretation:")
    print("   - Flat bounds (Ford-Roman): Apply strict flat-spacetime QI")
    print("   - Curved bounds (toy):      Assume curvature relaxes QI (heuristic)")
    print("   - Hybrid:                   Conservative estimate (most restrictive)")
    print("   - 4D proxy:                 Approximates full spacetime integral")
    
    print("\n" + "=" * 80)
    print("Phase E Extensions Summary")
    print("=" * 80)
    print("✓ 4D proxy integral: Extends 1+1D → 3+1D via spherical volume approximation")
    print("✓ Normalized margin: Δ̄ = (I-B)/|B| provides relative violation metric")
    print("✓ Bound family:      User-selectable models for sensitivity analysis")
    print("\nThese extensions allow exploration of bound model sensitivity and")
    print("dimensionality effects without claiming physical rigor for curved QI bounds.")
    print("\nIMPORTANT: Curved-toy bounds are heuristic placeholders. Rigorous curved-")
    print("spacetime QI inequalities remain an open research question in QFT on")
    print("curved backgrounds (especially for exotic Alcubierre metrics).")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_phase_e_demo()
