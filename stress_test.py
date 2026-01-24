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
from typing import Any, Dict, List, Tuple

import numpy as np
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False


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


def _fragility_sweep(
    param_name: str,
    param_values: np.ndarray,
    base_config: Dict[str, Any],
    trials: int,
    sigma_noise: float,
    rng: np.random.Generator,
) -> Tuple[List[float], List[float]]:
    """Sweep a parameter and compute fragility D at each value.
    
    Returns:
        (param_vals, D_vals) lists
    """
    from src.warp_qft.enhancement_pathway import EnhancementConfig
    from src.warp_qft.enhancement_pipeline import PipelineConfig, WarpBubbleEnhancementPipeline

    D_vals = []
    param_vals_out = []

    for pval in param_values:
        config = dict(base_config)
        config[param_name] = float(pval)

        energies = []
        for _ in range(trials):
            noisy = dict(config)
            for k in ["mu", "R", "Q", "squeezing_db"]:
                noisy[k] = float(noisy[k]) * (1.0 + rng.normal(0.0, sigma_noise))
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

        energies_arr = np.array(energies, dtype=float)
        mean_E = float(np.nanmean(energies_arr))
        std_E = float(np.nanstd(energies_arr))
        D = float(std_E / mean_E) if abs(mean_E) > 1e-12 else float("inf")

        if np.isfinite(D):
            D_vals.append(D)
            param_vals_out.append(float(pval))

    return param_vals_out, D_vals


def _fit_exponential(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """Fit y = a * exp(b * x) using log-linear regression.
    
    Returns:
        (a, b, r_squared)
    """
    x_arr = np.array(x)
    y_arr = np.array(y)

    # Filter positive y for log fit
    mask = y_arr > 0
    if not np.any(mask):
        return (np.nan, np.nan, np.nan)

    x_fit = x_arr[mask]
    y_fit = y_arr[mask]

    log_y = np.log(y_fit)
    # Linear regression on log(y) = log(a) + b*x
    coeffs = np.polyfit(x_fit, log_y, 1)
    b = coeffs[0]
    log_a = coeffs[1]
    a = np.exp(log_a)

    # R^2
    y_pred = a * np.exp(b * x_fit)
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return float(a), float(b), float(r_squared)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--sigma-noise", type=float, default=0.05, help="Stddev of multiplicative noise")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--fragility-fit", action="store_true", help="Run fragility(mu) sweep and fit D(mu)=a*e^(b*mu)")
    p.add_argument("--mu-sweep", type=str, default="0.05:0.60:8", help="mu sweep range (start:stop:count)")
    p.add_argument("--save-plots", action="store_true", help="Save fragility fit plot")

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
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # If fragility fit mode, run parameter sweep instead of edge cases
    if args.fragility_fit:
        parts = args.mu_sweep.split(":")
        if len(parts) != 3:
            print("--mu-sweep must be start:stop:count")
            return 2
        mu_start, mu_stop, mu_count = float(parts[0]), float(parts[1]), int(parts[2])
        mu_values = np.linspace(mu_start, mu_stop, mu_count)

        base_config = {"mu": 0.3, "R": 2.3, "Q": 1e6, "squeezing_db": 15.0, "num_bubbles": 3}

        print(f"Running fragility sweep over mu={mu_start}..{mu_stop} ({mu_count} points, {args.trials} trials each)...")
        mu_vals, D_vals = _fragility_sweep("mu", mu_values, base_config, args.trials, args.sigma_noise, rng)

        if len(mu_vals) < 3:
            print("Not enough valid points for fit (need >= 3)")
            return 2

        a, b, r2 = _fit_exponential(mu_vals, D_vals)

        fit_report = {
            "timestamp": _timestamp(),
            "command_args": vars(args),
            "parameter": "mu",
            "sweep_range": {"start": float(mu_start), "stop": float(mu_stop), "count": int(mu_count)},
            "base_config": base_config,
            "data": {"mu": mu_vals, "D": D_vals},
            "fit": {
                "model": "D(mu) = a * exp(b * mu)",
                "a": a,
                "b": b,
                "r_squared": r2,
            },
            "notes": "Fragility D = std(E)/mean(E); exponential fit via log-linear regression.",
        }

        if args.save_results:
            out_path = results_dir / f"fragility_fit_{fit_report['timestamp']}.json"
            out_path.write_text(json.dumps(_json_safe(fit_report), indent=2), encoding="utf-8")
            print(f"Wrote {out_path}")

        if args.save_plots and HAS_PLT:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(mu_vals, D_vals, label="Data", s=50, alpha=0.7)
            mu_fit = np.linspace(min(mu_vals), max(mu_vals), 100)
            D_fit = a * np.exp(b * mu_fit)
            ax.plot(mu_fit, D_fit, "r--", label=f"Fit: D={a:.3f}*exp({b:.3f}*μ), R²={r2:.3f}")
            ax.set_xlabel("Polymer parameter μ")
            ax.set_ylabel("Fragility D (std/mean)")
            ax.set_title("Fragility vs Polymer Parameter")
            ax.axhline(0.1, color="gray", linestyle=":", alpha=0.5, label="D=0.1 threshold")
            ax.legend()
            ax.grid(alpha=0.3)

            plot_path = results_dir / f"fragility_fit_{fit_report['timestamp']}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved plot: {plot_path}")

        print(f"\nFragility fit results:")
        print(f"  Model: D(mu) = {a:.4f} * exp({b:.4f} * mu)")
        print(f"  R²: {r2:.4f}")
        print(f"  Interpretation: {'Fragility increases exponentially with mu' if b > 0 else 'Fragility decreases with mu'}")

        return 0
    # “Edges” are intentionally provocative; treat outcomes as robustness signals.
    # Expanded to cover diverse extreme regimes: high/low mu, extreme Q, aggressive squeezing,
    # minimal/maximal enhancements, and mixed configurations.
    edge_cases: List[Dict[str, Any]] = [
        # Original cases (baseline)
        {"mu": 0.60, "R": 12.0, "Q": 5e3, "squeezing_db": 20.0, "num_bubbles": 4, "label": "high-mu-large-R"},
        {"mu": 0.01, "R": 0.50, "Q": 1e8, "squeezing_db": 5.0, "num_bubbles": 3, "label": "low-mu-tiny-R-high-Q"},
        {"mu": 0.20, "R": 10.0, "Q": 1e4, "squeezing_db": 25.0, "num_bubbles": 6, "label": "large-R-extreme-squeezing"},
        
        # Extreme mu regimes
        {"mu": 0.90, "R": 5.0, "Q": 1e6, "squeezing_db": 15.0, "num_bubbles": 3, "label": "extreme-mu"},
        {"mu": 0.005, "R": 2.0, "Q": 1e6, "squeezing_db": 15.0, "num_bubbles": 3, "label": "minimal-mu"},
        
        # Cavity Q extremes
        {"mu": 0.3, "R": 2.3, "Q": 1e10, "squeezing_db": 10.0, "num_bubbles": 2, "label": "ultra-high-Q"},
        {"mu": 0.3, "R": 2.3, "Q": 1e3, "squeezing_db": 15.0, "num_bubbles": 3, "label": "minimal-Q"},
        
        # Squeezing extremes
        {"mu": 0.3, "R": 2.3, "Q": 1e6, "squeezing_db": 30.0, "num_bubbles": 3, "label": "maximal-squeezing"},
        {"mu": 0.3, "R": 2.3, "Q": 1e6, "squeezing_db": 1.0, "num_bubbles": 3, "label": "minimal-squeezing"},
        
        # Multi-bubble extremes
        {"mu": 0.3, "R": 2.3, "Q": 1e6, "squeezing_db": 15.0, "num_bubbles": 10, "label": "many-bubbles"},
        {"mu": 0.3, "R": 2.3, "Q": 1e6, "squeezing_db": 15.0, "num_bubbles": 1, "label": "single-bubble"},
        
        # Conservative baseline (all minimal enhancements)
        {"mu": 0.05, "R": 2.0, "Q": 1e4, "squeezing_db": 5.0, "num_bubbles": 1, "label": "conservative-all"},
        
        # Aggressive baseline (all maximal enhancements)
        {"mu": 0.8, "R": 15.0, "Q": 1e9, "squeezing_db": 25.0, "num_bubbles": 8, "label": "aggressive-all"},
        
        # Mixed regimes (high in some parameters, low in others)
        {"mu": 0.7, "R": 1.0, "Q": 5e4, "squeezing_db": 8.0, "num_bubbles": 2, "label": "high-mu-small-R"},
        {"mu": 0.05, "R": 20.0, "Q": 1e8, "squeezing_db": 20.0, "num_bubbles": 7, "label": "low-mu-large-R-high-enhancements"},
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
                "label": base_params.get("label", "unlabeled"),
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

    if args.save_results:
        out_path = results_dir / f"stress_tests_{report['timestamp']}.json"
        out_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    # Console summary
    print(f"\nStress test summary ({len(report['edge_cases'])} edge cases):")
    for item in report["edge_cases"]:
        base = item["base_params"]
        stats = item["final_energy_stats"]
        label = item.get("label", "unlabeled")
        print(
            f"  [{label:30s}] mu={base['mu']:.2f}, R={base['R']:.1f}, Q={base['Q']:.1e}, "
            f"sq={base['squeezing_db']:.0f}dB, N={base['num_bubbles']} -> "
            f"D={stats['robustness_D']:.3f}, feasible={stats['feasible_rate']:.1%} ({item['interpretation']})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
