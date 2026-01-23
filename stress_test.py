#!/usr/bin/env python3
"""Edge-case stress testing for the enhanced pipeline.

Runs a small set of extreme configurations and perturbs parameters with
multiplicative Gaussian noise to estimate robustness of the final energy
requirement metric.

Outputs JSON to results/<session>/.

Usage:
  python stress_test.py --save-results --results-dir results/final_integration
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


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


def _run_pipeline(params: Dict[str, Any]) -> Dict[str, Any]:
    from src.warp_qft.enhancement_pathway import EnhancementConfig
    from src.warp_qft.enhancement_pipeline import PipelineConfig, WarpBubbleEnhancementPipeline

    enh = EnhancementConfig(
        cavity_Q=float(params["Q"]),
        squeezing_db=float(params["squeezing_db"]),
        num_bubbles=int(params["num_bubbles"]),
    )

    config = PipelineConfig(
        enhancement_config=enh,
        use_backreaction=True,
        backreaction_iterative=bool(params.get("backreaction_iterative", True)),
        backreaction_outer_iterations=int(params.get("backreaction_outer_iters", 3)),
        backreaction_relative_energy_tolerance=float(params.get("backreaction_rel_tol", 1e-4)),
        grid_resolution=10,
    )

    pipeline = WarpBubbleEnhancementPipeline(config)
    base = pipeline.compute_base_energy_requirement(float(params["mu"]), float(params["R"]))
    corr = pipeline.apply_all_corrections(base, float(params["mu"]), float(params["R"]))

    return {
        "base_energy": float(base),
        "final_energy": float(corr.get("final_energy", np.nan)),
        "feasible": bool(float(corr.get("final_energy", np.inf)) <= 1.0),
        "total_reduction_factor": float(corr.get("total_reduction_factor", np.nan)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--sigma-noise", type=float, default=0.05, help="Stddev of multiplicative noise")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--save-results", action="store_true")

    args = p.parse_args()

    # Ensure repo root is on sys.path (namespace package `src.*`).
    import sys
    from pathlib import Path

    repo_root = Path(__file__).parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    rng = np.random.default_rng(args.seed)

    # “Edges” are intentionally provocative; treat outcomes as robustness signals.
    edge_cases: List[Dict[str, Any]] = [
        {"mu": 0.60, "R": 12.0, "Q": 5e3, "squeezing_db": 20.0, "num_bubbles": 4},
        {"mu": 0.01, "R": 0.50, "Q": 1e8, "squeezing_db": 5.0, "num_bubbles": 3},
        {"mu": 0.20, "R": 10.0, "Q": 1e4, "squeezing_db": 25.0, "num_bubbles": 6},
    ]

    results: List[Dict[str, Any]] = []

    for base_params in edge_cases:
        energies: List[float] = []
        feasible_count = 0

        for _ in range(args.trials):
            noisy = dict(base_params)
            for k in ["mu", "R", "Q", "squeezing_db"]:
                noisy[k] = float(noisy[k]) * (1.0 + rng.normal(0.0, args.sigma_noise))
                # Keep parameters in sane ranges
                if k == "mu":
                    noisy[k] = float(np.clip(noisy[k], 1e-4, 2.0))
                if k == "R":
                    noisy[k] = float(np.clip(noisy[k], 1e-3, 50.0))
                if k == "Q":
                    noisy[k] = float(np.clip(noisy[k], 1e2, 1e12))
                if k == "squeezing_db":
                    noisy[k] = float(np.clip(noisy[k], 0.0, 60.0))

            out = _run_pipeline(noisy)
            energies.append(float(out["final_energy"]))
            feasible_count += 1 if out["feasible"] else 0

        energies_arr = np.array(energies, dtype=float)
        mean_E = float(np.nanmean(energies_arr))
        std_E = float(np.nanstd(energies_arr))
        D = float(std_E / mean_E) if abs(mean_E) > 1e-12 else float("inf")

        results.append(
            {
                "base_params": base_params,
                "trials": int(args.trials),
                "sigma_noise": float(args.sigma_noise),
                "final_energy_stats": {
                    "mean": mean_E,
                    "std": std_E,
                    "robustness_D": D,
                    "feasible_rate": float(feasible_count / max(1, args.trials)),
                },
                "interpretation": "fragile" if D > 0.1 else "robust",
            }
        )

    report = {
        "timestamp": _timestamp(),
        "command_args": vars(args),
        "edge_cases": results,
        "notes": "Robustness D = std(E)/mean(E) over multiplicative-noise trials.",
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.save_results:
        out_path = results_dir / f"stress_tests_{report['timestamp']}.json"
        out_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    # Console summary
    print("\nStress test summary:")
    for item in report["edge_cases"]:
        base = item["base_params"]
        stats = item["final_energy_stats"]
        print(
            f"  mu={base['mu']}, R={base['R']}, Q={base['Q']:.2g}, sq={base['squeezing_db']}dB, N={base['num_bubbles']} -> "
            f"D={stats['robustness_D']:.3f}, feasible_rate={stats['feasible_rate']:.2f} ({item['interpretation']})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
