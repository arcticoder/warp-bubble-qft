#!/usr/bin/env python3
"""End-to-end full-system integration runner.

This script is intentionally conservative: it runs a parameter grid over the
existing pipeline, then (optionally) runs toy QI+3D checks for each point.

Outputs a single JSON artifact into results/<session>/.

Usage:
  python full_integration.py --save-results --results-dir results/final_integration

Notes:
- This is a computational integration/consistency check, not a physics proof.
- QI bounds used in integrated checks are toy/heuristic; the report records
  which bound was used.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Keep imports local to avoid slowing down --help


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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def _build_grid(values: str) -> List[float]:
    """Parse either a comma list ('0.1,0.2') or a linspace ('0.05:0.3:5')."""
    raw = values.strip()
    if ":" in raw:
        parts = raw.split(":")
        if len(parts) != 3:
            raise ValueError("grid must be comma list or start:stop:count")
        start, stop, count = float(parts[0]), float(parts[1]), int(parts[2])
        return [float(x) for x in np.linspace(start, stop, count)]
    return [float(x) for x in raw.split(",") if x.strip()]


def run_point(
    *,
    mu: float,
    R: float,
    cavity_Q: float,
    squeezing_db: float,
    num_bubbles: int,
    backreaction_iterative: bool,
    backreaction_outer_iters: int,
    backreaction_rel_tol: float,
    include_qi_3d: bool,
    qi_3d_grid: int,
    qi_3d_t_final: float,
) -> Dict[str, Any]:
    from src.warp_qft.enhancement_pathway import EnhancementConfig
    from src.warp_qft.enhancement_pipeline import PipelineConfig, WarpBubbleEnhancementPipeline

    enh = EnhancementConfig(
        cavity_Q=cavity_Q,
        squeezing_db=squeezing_db,
        num_bubbles=num_bubbles,
    )

    config = PipelineConfig(
        enhancement_config=enh,
        use_backreaction=True,
        backreaction_iterative=backreaction_iterative,
        backreaction_outer_iterations=backreaction_outer_iters,
        backreaction_relative_energy_tolerance=backreaction_rel_tol,
        grid_resolution=10,
    )

    pipeline = WarpBubbleEnhancementPipeline(config)
    base_energy = pipeline.compute_base_energy_requirement(mu, R)
    corrections = pipeline.apply_all_corrections(base_energy, mu, R)

    enh_breakdown = corrections.get("enhancements", {})
    cavity_factor = float(enh_breakdown.get("cavity_enhancement", 1.0))
    squeezing_factor = float(enh_breakdown.get("squeezing_enhancement", 1.0))
    bubble_factor = float(enh_breakdown.get("multi_bubble_enhancement", enh_breakdown.get("bubble_enhancement", 1.0)))
    eta = float(cavity_factor * squeezing_factor * bubble_factor)

    out: Dict[str, Any] = {
        "inputs": {
            "mu": float(mu),
            "R": float(R),
            "cavity_Q": float(cavity_Q),
            "squeezing_db": float(squeezing_db),
            "num_bubbles": int(num_bubbles),
            "backreaction_iterative": bool(backreaction_iterative),
            "backreaction_outer_iters": int(backreaction_outer_iters),
            "backreaction_rel_tol": float(backreaction_rel_tol),
        },
        "pipeline": {
            "base_energy": float(base_energy),
            "final_energy": float(corrections.get("final_energy", np.nan)),
            "feasible": bool(float(corrections.get("final_energy", np.inf)) <= 1.0),
            "total_reduction_factor": float(corrections.get("total_reduction_factor", np.nan)),
            "eta": eta,
            "enhancement_factors": {
                "cavity": cavity_factor,
                "squeezing": squeezing_factor,
                "multi_bubble": bubble_factor,
                "total_reported": float(enh_breakdown.get("total_enhancement", eta)),
            },
        },
    }

    if include_qi_3d:
        # Use existing integrated runner as the canonical “toy combined” metric.
        from integrated_qi_3d_verification import run_3d_evolution
        from integrated_qi_3d_verification import alcubierre_metric_components
        from integrated_qi_3d_verification import curved_qi_integral

        # Recompute the same QI quantities used by integrated_qi_3d_verification.py
        n_samples = 100
        r_samples = np.linspace(0, 3 * R, n_samples)
        rho_base = -np.exp(-(r_samples**2) / (R**2))
        if mu > 0:
            arg = mu * np.abs(rho_base)
            with np.errstate(divide="ignore", invalid="ignore"):
                polymer_factor = np.where(arg < 1e-8, 1.0 - arg**2 / 6.0, np.sin(arg) / arg)
        else:
            polymer_factor = 1.0
        rho_samples = rho_base * polymer_factor

        dt_qi = 1.0 / n_samples
        qi_flat = float(np.trapezoid(rho_samples, dx=dt_qi))
        g_tt_curved, _ = alcubierre_metric_components(r_samples, R)
        qi_curved = curved_qi_integral(rho_samples, g_tt_curved, dt_qi)

        bound_flat = float(-1.0 / (16 * np.pi**2 * 1.0**2))
        bound_curved = float(-1.0 / (R**2))

        evolution = run_3d_evolution(
            N=qi_3d_grid,
            L=5.0,
            mu=mu,
            R=R,
            t_final=qi_3d_t_final,
            dt=0.001,
            polymer_enabled=True,
        )

        out["toy_qi_3d"] = {
            "qi": {
                "flat": {
                    "integral": qi_flat,
                    "bound": bound_flat,
                    "margin": float(qi_flat - bound_flat),
                    "violates": bool(qi_flat < bound_flat),
                },
                "curved": {
                    "integral": qi_curved,
                    "bound": bound_curved,
                    "margin": float(qi_curved - bound_curved),
                    "violates": bool(qi_curved < bound_curved),
                },
                "notes": "Toy/heuristic bounds; for integration regression checks only.",
            },
            "evolution": {
                "lyapunov_exponent": float(evolution.get("lyapunov_exponent", 0.0)),
                "growth_factor": float(evolution.get("growth_factor", 1.0)),
                "stable": bool(evolution.get("stable", False)),
                "steps": int(evolution.get("steps", 0)),
            },
        }

    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mu-grid", type=str, default="0.05:0.30:5", help="Comma list or start:stop:count")
    p.add_argument("--Q-grid", type=str, default="1e5,1e6,1e7", help="Comma list or start:stop:count")
    p.add_argument("--squeezing-grid", type=str, default="10,15,20", help="Comma list or start:stop:count")
    p.add_argument("--bubbles", type=str, default="3,4", help="Comma list")
    p.add_argument("--R", type=float, default=2.3, help="Bubble radius R")

    p.add_argument("--backreaction-iterative", action="store_true")
    p.add_argument("--backreaction-outer-iters", type=int, default=3)
    p.add_argument("--backreaction-rel-tol", type=float, default=1e-4)

    p.add_argument("--include-qi-3d", action="store_true", help="Run toy integrated QI+3D checks")
    p.add_argument("--qi-3d-grid", type=int, default=16)
    p.add_argument("--qi-3d-t-final", type=float, default=0.2)

    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--save-results", action="store_true")

    args = p.parse_args()

    # Ensure local imports work consistently.
    # The repo uses a namespace-package layout: `src.warp_qft` resolves when the
    # repository root is on sys.path.
    import sys
    from pathlib import Path

    repo_root = Path(__file__).parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    mu_vals = _build_grid(args.mu_grid)
    Q_vals = _build_grid(args.Q_grid)
    sq_vals = _build_grid(args.squeezing_grid)
    bubble_vals = [int(x) for x in args.bubbles.split(",") if x.strip()]

    runs: List[Dict[str, Any]] = []

    for mu in mu_vals:
        for Q in Q_vals:
            for sq_db in sq_vals:
                for n_bub in bubble_vals:
                    runs.append(
                        run_point(
                            mu=mu,
                            R=args.R,
                            cavity_Q=Q,
                            squeezing_db=sq_db,
                            num_bubbles=n_bub,
                            backreaction_iterative=args.backreaction_iterative,
                            backreaction_outer_iters=args.backreaction_outer_iters,
                            backreaction_rel_tol=args.backreaction_rel_tol,
                            include_qi_3d=args.include_qi_3d,
                            qi_3d_grid=args.qi_3d_grid,
                            qi_3d_t_final=args.qi_3d_t_final,
                        )
                    )

    report = {
        "timestamp": _timestamp(),
        "command_args": vars(args),
        "summary": {
            "num_points": len(runs),
            "num_feasible": int(sum(1 for r in runs if r["pipeline"]["feasible"])),
            "min_final_energy": float(min(r["pipeline"]["final_energy"] for r in runs)),
        },
        "grid": {
            "mu": mu_vals,
            "Q": Q_vals,
            "squeezing_db": sq_vals,
            "num_bubbles": bubble_vals,
            "R": float(args.R),
        },
        "runs": runs,
    }

    if args.save_results:
        out_path = results_dir / f"full_integration_{report['timestamp']}.json"
        out_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    print("\nFull integration summary:")
    print(f"  points:   {report['summary']['num_points']}")
    print(f"  feasible: {report['summary']['num_feasible']}")
    print(f"  min E:    {report['summary']['min_final_energy']:.6g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
