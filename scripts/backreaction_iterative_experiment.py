#!/usr/bin/env python3
"""Iterative backreaction experiment (toy self-consistency).

Runs an outer-loop coupling where the stress-energy profile amplitude is scaled
in proportion to the current energy estimate, then solves the (simplified)
Einstein equations to update an effective backreaction reduction factor.

Outputs timestamped JSON (and optional plots) into results/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.warp_qft.backreaction_solver import apply_backreaction_correction_iterative
from src.warp_qft.lqg_profiles import polymer_field_profile


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


def _summarize_solution(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    sol = diagnostics.get("last_solution")
    if not isinstance(sol, dict):
        return {}

    g_tt = sol.get("g_tt")
    g_rr = sol.get("g_rr")

    summary: Dict[str, Any] = {
        "metric_converged": bool(sol.get("converged", False)),
        "metric_iterations": int(sol.get("iterations", 0)),
        "metric_final_error": float(sol.get("final_error", float("inf"))),
    }

    if isinstance(g_tt, np.ndarray) and g_tt.size:
        summary.update(
            {
                "g_tt_min": float(np.min(g_tt)),
                "g_tt_max": float(np.max(g_tt)),
            }
        )
    if isinstance(g_rr, np.ndarray) and g_rr.size:
        summary.update(
            {
                "g_rr_min": float(np.min(g_rr)),
                "g_rr_max": float(np.max(g_rr)),
            }
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu", type=float, default=0.10)
    parser.add_argument("--R", type=float, default=2.3)
    parser.add_argument("--energy", type=float, default=1.0, help="Dimensionless base energy ratio to correct")

    parser.add_argument("--grid-size", type=int, default=400)
    parser.add_argument("--inner-iters", type=int, default=50)
    parser.add_argument("--outer-iters", type=int, default=10)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    parser.add_argument("--damping", type=float, default=0.7, help="Damping factor (0 < β < 1) for stability")
    parser.add_argument("--adaptive-damping", action="store_true", help="Enable adaptive damping schedule per outer iteration")
    parser.add_argument("--damping-beta0", type=float, default=None, help="Base damping β0 for adaptive schedule (defaults to --damping)")
    parser.add_argument("--damping-alpha", type=float, default=0.25, help="Adaptive damping sensitivity α")
    parser.add_argument("--damping-min", type=float, default=0.05, help="Lower clamp for adaptive damping")
    parser.add_argument("--damping-max", type=float, default=0.95, help="Upper clamp for adaptive damping")
    parser.add_argument("--regularization", type=float, default=1e-3, help="L2 regularization λ to bound norms")

    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    parser.add_argument(
        "--include-solution",
        action="store_true",
        help="Include full metric arrays in JSON (can be large)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    def rho_profile(r: np.ndarray) -> np.ndarray:
        return polymer_field_profile(r, args.mu, args.R)

    corrected, diagnostics = apply_backreaction_correction_iterative(
        float(args.energy),
        float(args.R),
        rho_profile,
        grid_size=int(args.grid_size),
        max_inner_iterations=int(args.inner_iters),
        max_outer_iterations=int(args.outer_iters),
        relative_energy_tolerance=float(args.rel_tol),
        damping_factor=float(args.damping),
        adaptive_damping=bool(args.adaptive_damping),
        damping_beta0=args.damping_beta0,
        damping_alpha=float(args.damping_alpha),
        damping_min=float(args.damping_min),
        damping_max=float(args.damping_max),
        regularization_lambda=float(args.regularization),
    )

    run = {
        "timestamp": _timestamp(),
        "params": {
            "mu": float(args.mu),
            "R": float(args.R),
            "energy_in": float(args.energy),
            "energy_out": float(corrected),
        },
        "diagnostics": diagnostics,
        "solution_summary": _summarize_solution(diagnostics),
    }

    if not args.include_solution:
        # Drop full arrays for routine archiving.
        run["diagnostics"].pop("last_solution", None)

    if args.save_results:
        out_path = results_dir / f"backreaction_iterative_{run['timestamp']}.json"
        out_path.write_text(json.dumps(_json_safe(run), indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    if args.save_plots:
        try:
            import matplotlib.pyplot as plt

            history = diagnostics.get("history", [])
            iters = [h.get("outer_iteration") for h in history]
            energies = [h.get("energy_out") for h in history]

            plt.figure(figsize=(7, 4))
            plt.plot(iters, energies, marker="o")
            plt.xlabel("Outer iteration")
            plt.ylabel("Energy (dimensionless)")
            plt.title("Iterative backreaction coupling (toy)")
            plt.grid(True, alpha=0.3)
            plot_path = results_dir / f"backreaction_iterative_{run['timestamp']}.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=160)
            plt.close()
            print(f"Wrote {plot_path}")
        except Exception as exc:
            print(f"Plotting failed: {exc}")

    # Always print a short summary.
    diverged_flag = " [DIVERGED]" if diagnostics.get("divergence_detected", False) else ""
    print(f"Energy: {args.energy:.6g} -> {corrected:.6g} (factor {corrected/args.energy if args.energy else float('nan'):.6g}){diverged_flag}")
    if diagnostics.get("divergence_detected"):
        print(f"  ⚠ {diagnostics.get('stability_note', 'Stability issue detected')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
