#!/usr/bin/env python3
"""
Synergy Integration Demo

Demonstrates the synergy factor S = exp(Σγ_ij) - 1 implementation and
integration with enhancement pathways and 3D evolution.

Usage:
    python demos/demo_synergy_integration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.warp_qft.enhancement_pathway import (
    EnhancementPathwayOrchestrator,
    EnhancementConfig,
)
from src.warp_qft.synergy import (
    SynergyCalculator,
    create_baseline_config,
    create_conservative_config,
    create_aggressive_config,
)


def main() -> int:
    print("\n" + "="*70)
    print("  Synergy Integration Demo")
    print("="*70)
    
    # Parameters
    base_energy = 1.0
    cavity_Q = 1e6
    squeezing_db = 15.0
    num_bubbles = 3
    
    print(f"\nEnhancement parameters:")
    print(f"  Cavity Q: {cavity_Q:.0e}")
    print(f"  Squeezing: {squeezing_db} dB")
    print(f"  Bubbles: {num_bubbles}")
    print(f"  Base energy: {base_energy}")
    
    # Test 1: Baseline (no synergy)
    print("\n" + "-"*70)
    print("  Test 1: Baseline (No Synergy)")
    print("-"*70)
    
    config_baseline = EnhancementConfig(
        cavity_Q=cavity_Q,
        squeezing_db=squeezing_db,
        num_bubbles=num_bubbles,
    )
    orch_baseline = EnhancementPathwayOrchestrator(config_baseline)
    result_baseline = orch_baseline.combine_all_enhancements(base_energy)
    
    print(f"Individual factors:")
    print(f"  Cavity: {result_baseline['cavity_enhancement']:.2f}×")
    print(f"  Squeezing: {result_baseline['squeezing_enhancement']:.2f}×")
    print(f"  Multi-bubble: {result_baseline['bubble_enhancement']:.2f}×")
    print(f"\nCombined enhancement:")
    print(f"  Multiplicative: {result_baseline['multiplicative_enhancement']:.2f}×")
    print(f"  Synergy factor S: {result_baseline['synergy_factor']:.4f}")
    print(f"  Total (with synergy): {result_baseline['total_enhancement']:.2f}×")
    print(f"  Energy: {base_energy:.2e} → {result_baseline['enhanced_energy']:.2e}")
    
    # Test 2: Conservative synergy
    print("\n" + "-"*70)
    print("  Test 2: Conservative Synergy")
    print("-"*70)
    
    config_conservative = EnhancementConfig(
        cavity_Q=cavity_Q,
        squeezing_db=squeezing_db,
        num_bubbles=num_bubbles,
        synergy_config=create_conservative_config(),
    )
    orch_conservative = EnhancementPathwayOrchestrator(config_conservative)
    result_conservative = orch_conservative.combine_all_enhancements(base_energy)
    
    print(f"Coupling coefficients (conservative):")
    synergy_cfg = config_conservative.synergy_config
    print(f"  γ_cavity,squeezing: {synergy_cfg.gamma_cavity_squeezing:.3f}")
    print(f"  γ_cavity,polymer: {synergy_cfg.gamma_cavity_polymer:.3f}")
    print(f"  γ_cavity,multi: {synergy_cfg.gamma_cavity_multi:.3f}")
    print(f"  γ_squeezing,polymer: {synergy_cfg.gamma_squeezing_polymer:.3f}")
    print(f"  γ_squeezing,multi: {synergy_cfg.gamma_squeezing_multi:.3f}")
    print(f"  γ_polymer,multi: {synergy_cfg.gamma_polymer_multi:.3f}")
    
    print(f"\nCombined enhancement:")
    print(f"  Multiplicative: {result_conservative['multiplicative_enhancement']:.2f}×")
    print(f"  Synergy factor S: {result_conservative['synergy_factor']:.4f}")
    print(f"  Synergy boost: {result_conservative['synergy_boost']:.3f}×")
    print(f"  Total (with synergy): {result_conservative['total_enhancement']:.2f}×")
    print(f"  Energy: {base_energy:.2e} → {result_conservative['enhanced_energy']:.2e}")
    
    # Test 3: Aggressive synergy
    print("\n" + "-"*70)
    print("  Test 3: Aggressive Synergy (Upper Bound)")
    print("-"*70)
    
    config_aggressive = EnhancementConfig(
        cavity_Q=cavity_Q,
        squeezing_db=squeezing_db,
        num_bubbles=num_bubbles,
        synergy_config=create_aggressive_config(),
    )
    orch_aggressive = EnhancementPathwayOrchestrator(config_aggressive)
    result_aggressive = orch_aggressive.combine_all_enhancements(base_energy)
    
    print(f"Coupling coefficients (aggressive):")
    synergy_cfg = config_aggressive.synergy_config
    print(f"  γ_cavity,squeezing: {synergy_cfg.gamma_cavity_squeezing:.3f}")
    print(f"  γ_cavity,polymer: {synergy_cfg.gamma_cavity_polymer:.3f}")
    print(f"  γ_cavity,multi: {synergy_cfg.gamma_cavity_multi:.3f}")
    print(f"  γ_squeezing,polymer: {synergy_cfg.gamma_squeezing_polymer:.3f}")
    print(f"  γ_squeezing,multi: {synergy_cfg.gamma_squeezing_multi:.3f}")
    print(f"  γ_polymer,multi: {synergy_cfg.gamma_polymer_multi:.3f}")
    
    print(f"\nCombined enhancement:")
    print(f"  Multiplicative: {result_aggressive['multiplicative_enhancement']:.2f}×")
    print(f"  Synergy factor S: {result_aggressive['synergy_factor']:.4f}")
    print(f"  Synergy boost: {result_aggressive['synergy_boost']:.3f}×")
    print(f"  Total (with synergy): {result_aggressive['total_enhancement']:.2f}×")
    print(f"  Energy: {base_energy:.2e} → {result_aggressive['enhanced_energy']:.2e}")
    
    # Comparison summary
    print("\n" + "="*70)
    print("  Summary: Impact of Synergy Modeling")
    print("="*70)
    
    print(f"\n{'Mode':<20} {'Mult. (×)':<15} {'S':<12} {'Total (×)':<15} {'Boost':<10}")
    print("-"*70)
    print(f"{'Baseline':<20} {result_baseline['multiplicative_enhancement']:<15.2f} "
          f"{result_baseline['synergy_factor']:<12.4f} "
          f"{result_baseline['total_enhancement']:<15.2f} "
          f"{result_baseline.get('synergy_boost', 1.0):<10.3f}")
    print(f"{'Conservative':<20} {result_conservative['multiplicative_enhancement']:<15.2f} "
          f"{result_conservative['synergy_factor']:<12.4f} "
          f"{result_conservative['total_enhancement']:<15.2f} "
          f"{result_conservative['synergy_boost']:<10.3f}")
    print(f"{'Aggressive':<20} {result_aggressive['multiplicative_enhancement']:<15.2f} "
          f"{result_aggressive['synergy_factor']:<12.4f} "
          f"{result_aggressive['total_enhancement']:<15.2f} "
          f"{result_aggressive['synergy_boost']:<10.3f}")
    
    print("\n" + "="*70)
    print("  Interpretation")
    print("="*70)
    print("""
Synergy factor S quantifies non-linear cross-pathway coupling:
- S = 0 (baseline): Purely multiplicative enhancement (independent pathways)
- S > 0: Positive synergy from correlated enhancements
- Conservative: γ_ij ∈ [0.02, 0.05] → S ≈ 0.21 → 1.21× boost
- Aggressive: γ_ij ∈ [0.08, 0.15] → S ≈ 1.21 → 2.21× boost

**Important**: Synergy is a *model choice*, not a fundamental prediction.
Coupling coefficients γ_ij are heuristic and require experimental validation.
Treat as hypothesis for parameter-space exploration, not established fact.

For 3D evolution, use: python full_3d_evolution.py --synergy-factor S
where S is the desired synergy factor (e.g., --synergy-factor 0.21)
""")
    
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
