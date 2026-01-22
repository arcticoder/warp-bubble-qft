#!/usr/bin/env python3
"""
Curved-Spacetime Quantum Inequality Verification

Extends Ford-Roman QI checks from flat spacetime to curved Alcubierre background.

In curved spacetime, the quantum inequality becomes:
    ∫ ρ(τ) g_μν dτ^μ dτ^ν ≥ -C / R²

where:
- ρ(τ) is energy density along timelike curve
- g_μν is the metric tensor (Alcubierre warp bubble)
- R is local curvature radius
- C is a dimensionless constant

This script:
1. Loads Alcubierre metric from toy_evolution outputs
2. Computes metric-weighted QI integral
3. Compares to flat-space bound from verify_qi_energy_density.py
4. Determines if violation persists in curved background

Usage:
    python curved_qi_verification.py [--metric-file FILE] [--save-results] [--save-plots]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.warp_qft.lqg_profiles import polymer_field_profile


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_alcubierre_metric(metric_file: Path | None = None) -> Dict[str, np.ndarray]:
    """
    Load Alcubierre metric components from toy_evolution output.
    
    Returns:
        Dictionary with x grid, g_tt, g_rr metric components
    """
    if metric_file is None or not metric_file.exists():
        # Use toy Alcubierre profile if no file provided
        x = np.linspace(-5.0, 5.0, 500)
        R_bubble = 2.3
        v = 0.1  # Warp velocity
        
        # Alcubierre metric components (simplified, spherically symmetric approx)
        # g_tt = -(1 - v² f²), g_rr = 1 + f²
        # where f(r) is shape function
        r = np.abs(x)
        sigma = 0.5  # Thickness parameter
        f = np.tanh((R_bubble - r) / sigma)
        
        g_tt = -(1 - v**2 * f**2)
        g_rr = 1 + f**2
        
        return {
            "x": x,
            "g_tt": g_tt,
            "g_rr": g_rr,
            "R_bubble": R_bubble,
            "v_warp": v,
            "source": "toy_profile"
        }
    else:
        # Load from file
        data = json.load(metric_file.open())
        # Extract metric from toy_evolution output format
        x = np.array(data.get("x", []))
        # Placeholder: actual toy_evolution doesn't save metric yet
        # This would need integration with toy_evolution.py output
        raise NotImplementedError("Loading from toy_evolution output not yet implemented")


def compute_curvature_radius(g_tt: np.ndarray, g_rr: np.ndarray, x: np.ndarray) -> float:
    """
    Estimate local curvature radius from metric.
    
    For Alcubierre metric, R ~ characteristic scale where metric deviates from Minkowski.
    """
    dx = x[1] - x[0]
    
    # Compute Ricci scalar approximation (simplified)
    # R ≈ |g_μν - η_μν| / L²
    deviation_tt = np.abs(g_tt + 1.0)  # Deviation from Minkowski (-1)
    deviation_rr = np.abs(g_rr - 1.0)  # Deviation from Minkowski (+1)
    
    max_deviation = np.max(deviation_tt + deviation_rr)
    
    # Characteristic length scale
    L = np.ptp(x) / 10.0  # ~10% of domain
    
    # Curvature radius estimate
    if max_deviation > 1e-6:
        R_curv = L / np.sqrt(max_deviation)
    else:
        R_curv = float('inf')  # Nearly flat
    
    return float(R_curv)


def flat_qi_integral(rho_profile: np.ndarray, x: np.ndarray, sampling_width: float = 1.0) -> float:
    """
    Compute flat-space QI integral with Lorentzian sampling.
    
    ∫ ρ(x) w(x) dx, where w(x) = (Δ/π) / (x² + Δ²)
    """
    Delta = sampling_width
    w = (Delta / np.pi) / (x**2 + Delta**2)
    
    dx = x[1] - x[0]
    integral = np.sum(rho_profile * w) * dx
    
    return float(integral)


def curved_qi_integral(
    rho_profile: np.ndarray,
    g_tt: np.ndarray,
    g_rr: np.ndarray,
    x: np.ndarray,
    sampling_width: float = 1.0
) -> float:
    """
    Compute curved-space QI integral with metric weighting.
    
    ∫ ρ(τ) √|g| w(τ) dτ
    
    where √|g| is the metric volume element.
    """
    Delta = sampling_width
    w = (Delta / np.pi) / (x**2 + Delta**2)
    
    # Volume element: √|g| = √(|g_tt| * g_rr) in 1+1D
    sqrt_g = np.sqrt(np.abs(g_tt) * g_rr)
    
    dx = x[1] - x[0]
    integral = np.sum(rho_profile * sqrt_g * w) * dx
    
    return float(integral)


def ford_roman_bound(Delta_t: float, dimension: int = 4) -> float:
    """
    Ford-Roman quantum inequality bound for massless scalar.
    
    Flat space: I ≥ -C / (Δt)^d
    where d = 4 for 3+1D, d = 2 for 1+1D
    """
    C_FR = 1.0 / (16 * np.pi**2)  # Typical coefficient
    bound = -C_FR / (Delta_t ** dimension)
    return float(bound)


def curved_ford_roman_bound(R_curv: float, Delta_t: float) -> float:
    """
    Curved-space QI bound (heuristic extension).
    
    I ≥ -C / R²
    
    where R is curvature radius. In high-curvature regions (R small),
    bound becomes more restrictive.
    """
    C = 1.0  # Dimensionless constant (order unity)
    if R_curv < 1e-6:
        R_curv = 1e-6  # Avoid division by zero
    bound = -C / (R_curv**2)
    return float(bound)


def verify_curved_qi(
    mu: float = 0.3,
    R_bubble: float = 2.3,
    sampling_width: float = 1.0,
    metric_file: Path | None = None
) -> Dict[str, Any]:
    """
    Main verification routine for curved QI.
    
    Returns:
        Comprehensive diagnostics dict
    """
    # Load metric
    metric = load_alcubierre_metric(metric_file)
    x = metric["x"]
    g_tt = metric["g_tt"]
    g_rr = metric["g_rr"]
    
    # Compute energy density profile
    rho = polymer_field_profile(x, mu, R_bubble)
    
    # Compute integrals
    I_flat = flat_qi_integral(rho, x, sampling_width)
    I_curved = curved_qi_integral(rho, g_tt, g_rr, x, sampling_width)
    
    # Compute bounds
    R_curv = compute_curvature_radius(g_tt, g_rr, x)
    bound_flat = ford_roman_bound(sampling_width, dimension=2)  # 1+1D
    bound_curved = curved_ford_roman_bound(R_curv, sampling_width)
    
    # Violation checks
    violates_flat = I_flat < bound_flat
    violates_curved = I_curved < bound_curved
    
    # Metric effects
    metric_enhancement = I_curved / I_flat if abs(I_flat) > 1e-12 else 1.0
    
    return {
        "parameters": {
            "mu": float(mu),
            "R_bubble": float(R_bubble),
            "sampling_width": float(sampling_width),
        },
        "metric_info": {
            "source": metric.get("source", "unknown"),
            "curvature_radius": float(R_curv),
            "grid_size": int(len(x)),
            "x_min": float(x.min()),
            "x_max": float(x.max()),
        },
        "integrals": {
            "flat_space": float(I_flat),
            "curved_space": float(I_curved),
            "metric_enhancement_factor": float(metric_enhancement),
        },
        "bounds": {
            "ford_roman_flat": float(bound_flat),
            "curved_bound": float(bound_curved),
        },
        "violations": {
            "violates_flat_bound": bool(violates_flat),
            "violates_curved_bound": bool(violates_curved),
            "violation_margin_flat": float(I_flat - bound_flat),
            "violation_margin_curved": float(I_curved - bound_curved),
        },
        "energy_profile": {
            "peak_density": float(rho.min()),
            "total_integrated": float(np.sum(rho) * (x[1] - x[0])),
            "rms_density": float(np.sqrt(np.mean(rho**2))),
        },
        "interpretation": (
            "QI violation persists in curved space"
            if violates_curved
            else "No curved-space QI violation detected"
        ),
    }


def plot_curved_qi_comparison(
    result: Dict[str, Any],
    save_path: Path | None = None
) -> None:
    """
    Generate comparison plots: flat vs curved integrands, bounds.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Panel 1: Integrals and bounds
    ax = axes[0]
    I_flat = result["integrals"]["flat_space"]
    I_curved = result["integrals"]["curved_space"]
    B_flat = result["bounds"]["ford_roman_flat"]
    B_curved = result["bounds"]["curved_bound"]
    
    x_pos = np.arange(2)
    integrals = [I_flat, I_curved]
    bounds = [B_flat, B_curved]
    
    ax.bar(x_pos - 0.15, integrals, 0.3, label="QI Integral", alpha=0.8)
    ax.bar(x_pos + 0.15, bounds, 0.3, label="QI Bound", alpha=0.8)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Flat Space", "Curved Space"])
    ax.set_ylabel("Energy Integral")
    ax.set_title("Quantum Inequality: Flat vs Curved Spacetime")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Violation margins
    ax = axes[1]
    margins = [
        result["violations"]["violation_margin_flat"],
        result["violations"]["violation_margin_curved"]
    ]
    colors = ['red' if m < 0 else 'green' for m in margins]
    
    ax.bar(x_pos, margins, color=colors, alpha=0.7)
    ax.axhline(0, color='k', linestyle='-', linewidth=1.2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Flat Space", "Curved Space"])
    ax.set_ylabel("Violation Margin (Integral - Bound)")
    ax.set_title("QI Violation Status (negative = violation)")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"Saved plot: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify quantum inequalities in curved Alcubierre spacetime"
    )
    parser.add_argument("--mu", type=float, default=0.3, help="Polymer parameter μ")
    parser.add_argument("--R", type=float, default=2.3, help="Bubble radius")
    parser.add_argument("--sampling-width", type=float, default=1.0, help="Temporal sampling width Δt")
    parser.add_argument("--metric-file", type=str, help="Path to metric JSON (from toy_evolution)")
    parser.add_argument("--save-results", action="store_true", help="Save JSON results")
    parser.add_argument("--save-plots", action="store_true", help="Save plots")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    metric_file = Path(args.metric_file) if args.metric_file else None
    
    # Run verification
    result = verify_curved_qi(
        mu=args.mu,
        R_bubble=args.R,
        sampling_width=args.sampling_width,
        metric_file=metric_file
    )
    
    # Add metadata
    result["timestamp"] = _timestamp()
    result["command_args"] = vars(args)
    
    # Print summary
    print("\n" + "="*70)
    print("  Curved-Space Quantum Inequality Verification")
    print("="*70)
    print(f"\nParameters:")
    print(f"  μ = {result['parameters']['mu']:.3f}")
    print(f"  R_bubble = {result['parameters']['R_bubble']:.3f}")
    print(f"  Sampling width Δt = {result['parameters']['sampling_width']:.3f}")
    print(f"\nMetric:")
    print(f"  Source: {result['metric_info']['source']}")
    print(f"  Curvature radius R_curv = {result['metric_info']['curvature_radius']:.3f}")
    print(f"\nIntegrals:")
    print(f"  Flat space:   {result['integrals']['flat_space']:+.6e}")
    print(f"  Curved space: {result['integrals']['curved_space']:+.6e}")
    print(f"  Enhancement:  {result['integrals']['metric_enhancement_factor']:.3f}×")
    print(f"\nBounds:")
    print(f"  Ford-Roman (flat):  {result['bounds']['ford_roman_flat']:+.6e}")
    print(f"  Curved bound:       {result['bounds']['curved_bound']:+.6e}")
    print(f"\nViolations:")
    print(f"  Flat space:   {'✓ VIOLATES' if result['violations']['violates_flat_bound'] else '✗ no violation'}")
    print(f"  Curved space: {'✓ VIOLATES' if result['violations']['violates_curved_bound'] else '✗ no violation'}")
    print(f"\n{result['interpretation']}")
    print("="*70 + "\n")
    
    # Save results
    if args.save_results:
        out_file = results_dir / f"curved_qi_{result['timestamp']}.json"
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved: {out_file}")
    
    # Save plots
    if args.save_plots:
        plot_file = results_dir / f"curved_qi_{result['timestamp']}.png"
        plot_curved_qi_comparison(result, plot_file)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
