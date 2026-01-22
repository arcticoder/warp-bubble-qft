#!/usr/bin/env python3
"""Toy 1D evolution harness (explicitly not full 3+1 GR).

This script evolves a simple 1D metric-perturbation proxy h(r, t) via a
reaction-diffusion-style PDE driven by a negative energy density profile.

It exists to provide:
- A reproducible stability smoke test
- A place to hang coarse causality/signature screening
- Timestamped JSON/plot artifacts in results/

Non-claims:
- Not a constrained 3+1 evolution
- No gauge conditions or constraint damping
- No physically validated stress-energy closure
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.warp_qft.causality import screen_spherical_metric
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=0.10)
    p.add_argument("--R", type=float, default=2.3)
    p.add_argument("--domain-mult", type=float, default=3.0, help="Domain half-width as multiple of R")
    p.add_argument("--n", type=int, default=401, help="Grid points")

    p.add_argument("--t-final", type=float, default=1.0)
    p.add_argument("--dt", type=float, default=0.0, help="If 0, choose automatically")

    p.add_argument("--D", type=float, default=0.02, help="Diffusion coefficient")
    p.add_argument("--alpha", type=float, default=0.5, help="Drive strength from rho")
    p.add_argument("--gamma", type=float, default=0.2, help="Damping")
    p.add_argument("--epsilon", type=float, default=0.1, help="Metric coupling scale")

    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--save-results", action="store_true")
    p.add_argument("--save-plots", action="store_true")

    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ts = _timestamp()

    R_max = float(args.domain_mult) * float(args.R)
    r = np.linspace(-R_max, R_max, int(args.n))
    dr = float(r[1] - r[0])

    rho = polymer_field_profile(r, float(args.mu), float(args.R))

    h = np.zeros_like(r)

    dt = float(args.dt)
    if dt <= 0:
        # Simple explicit stability heuristic for diffusion.
        dt = 0.25 * dr * dr / max(float(args.D), 1e-12)
        dt = min(dt, 1e-2)

    steps = int(np.ceil(float(args.t_final) / dt))
    steps = min(steps, 5000)

    max_abs_h = []
    ok_flags = []

    for _ in range(steps):
        # Laplacian (1D second derivative)
        dh_dr = np.gradient(h, dr)
        d2h_dr2 = np.gradient(dh_dr, dr)

        rhs = float(args.D) * d2h_dr2 + float(args.alpha) * rho - float(args.gamma) * h
        h = h + dt * rhs

        g_tt = -1.0 + float(args.epsilon) * h
        g_rr = 1.0 + float(args.epsilon) * h

        screen = screen_spherical_metric(r, g_tt, g_rr)
        max_abs_h.append(float(np.max(np.abs(h))))
        ok_flags.append(bool(screen["ok"]))

        if not screen["ok"]:
            # Stop early on obvious pathology.
            break

    g_tt = -1.0 + float(args.epsilon) * h
    g_rr = 1.0 + float(args.epsilon) * h
    screen = screen_spherical_metric(r, g_tt, g_rr)

    out: Dict[str, Any] = {
        "timestamp": ts,
        "params": {
            "mu": float(args.mu),
            "R": float(args.R),
            "domain_mult": float(args.domain_mult),
            "n": int(args.n),
            "t_final": float(args.t_final),
            "dt": float(dt),
            "steps_executed": int(len(max_abs_h)),
            "D": float(args.D),
            "alpha": float(args.alpha),
            "gamma": float(args.gamma),
            "epsilon": float(args.epsilon),
        },
        "series": {
            "max_abs_h": max_abs_h,
            "screen_ok": ok_flags,
        },
        "final": {
            "screen": screen,
            "r": r,
            "rho": rho,
            "h": h,
            "g_tt": g_tt,
            "g_rr": g_rr,
        },
        "notes": {
            "scope": "toy",
            "non_claims": [
                "Not a constrained 3+1 GR evolution",
                "No gauge conditions or constraint damping",
                "Stress-energy closure is heuristic",
            ],
        },
    }

    if args.save_results:
        json_path = results_dir / f"toy_evolution_{ts}.json"
        json_path.write_text(json.dumps(_json_safe(out), indent=2), encoding="utf-8")
        print(f"Wrote {json_path}")

    if args.save_plots:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].plot(r, rho, label="rho(r)")
            ax[0].plot(r, h, label="h(r)")
            ax[0].set_title("Toy fields")
            ax[0].grid(True, alpha=0.3)
            ax[0].legend()

            ax[1].plot(r, g_tt, label="g_tt")
            ax[1].plot(r, g_rr, label="g_rr")
            ax[1].set_title("Toy metric components")
            ax[1].grid(True, alpha=0.3)
            ax[1].legend()

            fig.suptitle("Toy evolution (final state)")
            fig.tight_layout()

            png_path = results_dir / f"toy_evolution_{ts}.png"
            fig.savefig(png_path, dpi=160)
            plt.close(fig)
            print(f"Wrote {png_path}")
        except Exception as exc:
            print(f"Plotting failed: {exc}")

    print(f"Final screen ok: {screen['ok']} | g_tt in [{screen['g_tt_min']:.3g}, {screen['g_tt_max']:.3g}] | g_rr in [{screen['g_rr_min']:.3g}, {screen['g_rr_max']:.3g}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
