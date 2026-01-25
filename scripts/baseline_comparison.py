#!/usr/bin/env python3
"""Baseline comparison: isolate reduction factors.

Runs the pipeline with various toggles to decompose the total energy reduction
into individual contributions:
- Van den Broeck-Natário geometric reduction
- LQG polymer corrections
- Backreaction (quick vs full)
- Enhancement pathways (conservative vs aggressive)

Outputs a timestamped JSON comparison table showing how each factor contributes.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.warp_qft.enhancement_pipeline import (
    PipelineConfig,
    WarpBubbleEnhancementPipeline,
)
from src.warp_qft.enhancement_pathway import EnhancementConfig


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
    except Exception:
        pass
    return str(obj)


def run_configuration(
    label: str,
    mu: float,
    R: float,
    use_vdb: bool,
    use_backreaction: bool,
    backreaction_quick: bool,
    backreaction_iterative: bool,
    enhancement_config: EnhancementConfig,
) -> Dict[str, Any]:
    """Run pipeline with specific configuration and return energy breakdown."""
    config = PipelineConfig(
        mu_min=mu - 0.01,
        mu_max=mu + 0.01,
        R_min=R - 0.1,
        R_max=R + 0.1,
        use_vdb_natario=use_vdb,
        use_backreaction=use_backreaction,
        backreaction_quick=backreaction_quick,
        backreaction_iterative=backreaction_iterative,
        enhancement_config=enhancement_config,
        grid_resolution=10,  # Quick evaluation
    )

    pipeline = WarpBubbleEnhancementPipeline(config)

    # Compute base and corrected energy
    base_energy = pipeline.compute_base_energy_requirement(mu, R)
    corrections = pipeline.apply_all_corrections(base_energy, mu, R)

    result = {
        "label": label,
        "config": {
            "use_vdb_natario": use_vdb,
            "use_backreaction": use_backreaction,
            "backreaction_quick": backreaction_quick,
            "backreaction_iterative": backreaction_iterative,
            "cavity_Q": enhancement_config.cavity_Q,
            "squeezing_db": enhancement_config.squeezing_db,
            "num_bubbles": enhancement_config.num_bubbles,
        },
        "energies": {
            "base_energy": float(base_energy),
            "final_energy": float(corrections["final_energy"]),
            "reduction_factor": float(base_energy / corrections["final_energy"])
            if corrections["final_energy"] > 0
            else float("inf"),
        },
    }

    # Extract intermediate factors if available
    if "backreaction" in corrections:
        br_info = corrections["backreaction"]
        if isinstance(br_info, dict):
            result["backreaction_factor"] = br_info.get("reduction_factor", 1.0)

    if "enhancements" in corrections:
        enh = corrections["enhancements"]
        if isinstance(enh, dict):
            result["enhancement_factors"] = {
                "cavity": float(enh.get("cavity_enhancement", 1.0)),
                "squeezing": float(enh.get("squeezing_enhancement", 1.0)),
                "multi_bubble": float(enh.get("multi_bubble_enhancement", 1.0)),
                "total": float(enh.get("total_enhancement", 1.0)),
            }

    return result


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compare pipeline baselines to isolate reduction factors"
    )
    p.add_argument("--mu", type=float, default=0.10)
    p.add_argument("--R", type=float, default=2.3)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--save-results", action="store_true")

    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ts = _timestamp()
    mu = float(args.mu)
    R = float(args.R)

    print("=" * 70)
    print("BASELINE COMPARISON: Isolating Reduction Factors")
    print("=" * 70)
    print(f"Parameters: μ={mu:.3f}, R={R:.2f}")
    print()

    # Define enhancement configurations
    conservative_enh = EnhancementConfig(
        cavity_Q=1e4, squeezing_db=0.0, num_bubbles=1
    )
    standard_enh = EnhancementConfig(
        cavity_Q=1e6, squeezing_db=15.0, num_bubbles=3
    )

    configs = []

    # 1. Baseline: no VdB, no backreaction, no enhancements
    configs.append(
        (
            "1_baseline",
            mu,
            R,
            False,
            False,
            True,
            False,
            conservative_enh,
            "Baseline (no VdB, no BR, no enh)",
        )
    )

    # 2. Add VdB-Natário
    configs.append(
        (
            "2_vdb",
            mu,
            R,
            True,
            False,
            True,
            False,
            conservative_enh,
            "+ VdB-Natário geometric reduction",
        )
    )

    # 3. Add quick backreaction
    configs.append(
        (
            "3_vdb_br_quick",
            mu,
            R,
            True,
            True,
            True,
            False,
            conservative_enh,
            "+ quick backreaction (~15%)",
        )
    )

    # 4. Use full backreaction (not iterative, but not quick)
    configs.append(
        (
            "4_vdb_br_full",
            mu,
            R,
            True,
            True,
            False,
            False,
            conservative_enh,
            "+ full backreaction solve",
        )
    )

    # 5. Add conservative enhancements (already using conservative_enh)
    # This is same as 3, so skip or note it's the same

    # 6. Add standard enhancements
    configs.append(
        (
            "5_vdb_br_enh_std",
            mu,
            R,
            True,
            True,
            True,
            False,
            standard_enh,
            "+ standard enhancements (Q=1e6, sqz=15dB, N=3)",
        )
    )

    # 7. Iterative backreaction + standard enhancements
    configs.append(
        (
            "6_vdb_br_iter_enh",
            mu,
            R,
            True,
            True,
            False,
            True,
            standard_enh,
            "+ iterative backreaction + standard enh",
        )
    )

    results = []
    for (
        label,
        mu_val,
        R_val,
        use_vdb,
        use_br,
        br_quick,
        br_iter,
        enh_cfg,
        desc,
    ) in configs:
        print(f"Running: {desc}...")
        try:
            result = run_configuration(
                label, mu_val, R_val, use_vdb, use_br, br_quick, br_iter, enh_cfg
            )
            result["description"] = desc
            results.append(result)
            base = result["energies"]["base_energy"]
            final = result["energies"]["final_energy"]
            factor = result["energies"]["reduction_factor"]
            print(f"  Base: {base:.6f}, Final: {final:.6f}, Factor: {factor:.2f}×")
        except Exception as exc:
            print(f"  ⚠ Failed: {exc}")
            results.append(
                {
                    "label": label,
                    "description": desc,
                    "error": str(exc),
                }
            )
        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(
        f"{'Config':<30} {'Base':<12} {'Final':<12} {'Factor':<10} {'Reduction %':<12}"
    )
    print("-" * 70)

    for r in results:
        if "error" not in r:
            cfg = r["description"][:28]
            base = r["energies"]["base_energy"]
            final = r["energies"]["final_energy"]
            factor = r["energies"]["reduction_factor"]
            reduction_pct = (1 - 1 / factor) * 100 if factor > 1 else 0.0
            print(
                f"{cfg:<30} {base:<12.6f} {final:<12.6f} {factor:<10.2f} {reduction_pct:<12.1f}"
            )

    output = {
        "timestamp": ts,
        "parameters": {"mu": mu, "R": R},
        "configurations": results,
    }

    if args.save_results:
        out_path = results_dir / f"baseline_comparison_{ts}.json"
        out_path.write_text(json.dumps(_json_safe(output), indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")

    print()
    print("=" * 70)
    print("Key observations:")
    print("  - Compare config 1 vs 2: VdB-Natário contribution")
    print("  - Compare config 2 vs 3: Quick backreaction contribution")
    print("  - Compare config 3 vs 4: Quick vs full backreaction")
    print("  - Compare config 3 vs 5: Enhancement pathway contribution")
    print("  - Compare config 5 vs 6: Quick vs iterative backreaction + enh")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
