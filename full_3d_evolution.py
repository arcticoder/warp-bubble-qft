#!/usr/bin/env python3
"""
3+1D Stability Analysis with Polymer-Corrected ADM Evolution

Extends toy_evolution.py to quasi-3D using simplified ADM equations with
Loop Quantum Gravity polymer modifications to the extrinsic curvature.

**Scope**: This is a toy computational exploration, NOT full 3+1 GR:
- Uses simplified ADM evolution (no constraint damping, no gauge fixing)
- Polymer correction K_ij → sin(μ̄ K_ij)/μ̄ applied heuristically
- Stability assessed via Lyapunov exponent of metric norm growth
- 3D grid uses finite differences (no sophisticated discretization)

**Non-claims**:
- Not a constrained Hamiltonian evolution
- Not solving Einstein equations exactly
- Not validated against known GR solutions
- Results are qualitative stability indicators only

Usage:
    python full_3d_evolution.py [--grid-size N] [--t-final T] [--save-results] [--save-plots]
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
        # For large 3D arrays, save only summary statistics
        return {
            "shape": list(obj.shape),
            "mean": float(np.mean(obj)),
            "std": float(np.std(obj)),
            "min": float(np.min(obj)),
            "max": float(np.max(obj)),
        }
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def polymer_correction_K(K: np.ndarray, mu_bar: float = 0.1) -> np.ndarray:
    """
    Apply polymer modification to extrinsic curvature.
    
    K_ij → sin(μ̄ K_ij) / μ̄
    
    This is the holonomy correction from Loop Quantum Gravity.
    """
    if mu_bar < 1e-12:
        return K  # Classical limit
    
    K_poly = np.sin(mu_bar * K) / mu_bar
    return K_poly


def compute_adm_evolution_step(
    g_xx: np.ndarray,
    g_yy: np.ndarray,
    g_zz: np.ndarray,
    K_xx: np.ndarray,
    K_yy: np.ndarray,
    K_zz: np.ndarray,
    alpha: float,
    rho: np.ndarray,
    dx: float,
    dt: float,
    mu_bar: float = 0.1,
    polymer_enabled: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simplified ADM evolution step (no shift, diagonal metric, flat topology).
    
    Equations (highly simplified, NOT full Einstein):
        ∂_t g_ij = -2α K_ij
        ∂_t K_ij ≈ -∇_i ∇_j α + α (R_ij - 2K_ik K^k_j + K K_ij) + source
    
    With polymer: K → sin(μ̄K)/μ̄ before evolution
    
    Returns:
        Updated (g_xx, g_yy, g_zz, K_xx, K_yy, K_zz)
    """
    # Apply polymer correction to K_ij
    if polymer_enabled:
        K_xx_eff = polymer_correction_K(K_xx, mu_bar)
        K_yy_eff = polymer_correction_K(K_yy, mu_bar)
        K_zz_eff = polymer_correction_K(K_zz, mu_bar)
    else:
        K_xx_eff = K_xx
        K_yy_eff = K_yy
        K_zz_eff = K_zz
    
    # Metric evolution: ∂_t g_ij = -2α K_ij
    g_xx_new = g_xx - 2 * alpha * K_xx_eff * dt
    g_yy_new = g_yy - 2 * alpha * K_yy_eff * dt
    g_zz_new = g_zz - 2 * alpha * K_zz_eff * dt
    
    # Simplified K evolution (ignoring spatial derivatives for stability test)
    # Add damping and source from energy density
    trace_K = K_xx + K_yy + K_zz
    
    # Heuristic: K grows from stress-energy, decays from trace
    # This is NOT the real ADM equation, just a proxy for stability analysis
    K_source = -0.1 * rho  # Negative energy drives K growth
    K_damping = -0.05 * trace_K  # Trace damping
    
    K_xx_new = K_xx + (K_source + K_damping) * dt
    K_yy_new = K_yy + (K_source + K_damping) * dt
    K_zz_new = K_zz + (K_source + K_damping) * dt
    
    return g_xx_new, g_yy_new, g_zz_new, K_xx_new, K_yy_new, K_zz_new


def compute_metric_norm(g_xx: np.ndarray, g_yy: np.ndarray, g_zz: np.ndarray) -> float:
    """
    Compute a simple norm of the spatial metric: ||g|| = sqrt(mean(g_ii²))
    """
    norm_sq = np.mean(g_xx**2 + g_yy**2 + g_zz**2)
    return float(np.sqrt(norm_sq))


def compute_lyapunov_exponent(norm_history: list[float], dt: float) -> float:
    """
    Estimate Lyapunov exponent from metric norm growth.
    
    λ ≈ (1/T) log(||g(T)|| / ||g(0)||)
    
    Positive λ → exponential growth → unstable
    Negative λ → exponential decay → stable
    """
    if len(norm_history) < 2:
        return 0.0
    
    T = dt * (len(norm_history) - 1)
    norm_0 = norm_history[0]
    norm_T = norm_history[-1]
    
    if norm_0 < 1e-12 or norm_T < 1e-12:
        return 0.0
    
    lyap = (1.0 / T) * np.log(norm_T / norm_0)
    return float(lyap)


def evolve_3d_metric(
    grid_size: int = 32,
    domain_size: float = 5.0,
    t_final: float = 1.0,
    dt: float = 0.001,
    mu: float = 0.1,
    mu_bar: float = 0.1,
    R_bubble: float = 2.3,
    polymer_enabled: bool = True,
    synergy_factor: float = 0.0,
) -> Dict[str, Any]:
    """
    Main 3D evolution routine.
    
    Args:
        synergy_factor: Synergy enhancement S = exp(Σγ_ij) - 1 (default 0 = baseline)
    
    Returns:
        Dictionary with evolution history and diagnostics
    """
    # Setup 3D grid (for simplicity, use spherically symmetric initial data on Cartesian grid)
    x = np.linspace(-domain_size, domain_size, grid_size)
    y = np.linspace(-domain_size, domain_size, grid_size)
    z = np.linspace(-domain_size, domain_size, grid_size)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    dx = float(x[1] - x[0])
    
    # Radial coordinate (spherical symmetry approximation)
    r_3d = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Initial energy density profile (negative in bubble)
    rho_3d = np.zeros_like(r_3d)
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                r = r_3d[i, j, k]
                rho_3d[i, j, k] = polymer_field_profile(np.array([r]), mu, R_bubble)[0]
    
    # Synergy-modulated energy density: ρ_syn = ρ * (1 + S)
    # S = 0 (baseline) means no synergy, purely multiplicative enhancements
    rho_syn_3d = rho_3d * (1.0 + synergy_factor)
    
    # Use synergistic density in evolution (fallback to baseline if S=0)
    rho_effective = rho_syn_3d
    
    # Initial metric: nearly flat
    g_xx = np.ones_like(r_3d)
    g_yy = np.ones_like(r_3d)
    g_zz = np.ones_like(r_3d)
    
    # Initial extrinsic curvature: small perturbation
    K_xx = 0.01 * rho_3d
    K_yy = 0.01 * rho_3d
    K_zz = 0.01 * rho_3d
    
    # Lapse function (simplified, constant)
    alpha = 1.0
    
    # Evolution loop
    steps = int(t_final / dt)
    steps = min(steps, 1000)  # Cap to avoid excessive computation
    
    norm_history = []
    time_history = []
    
    for step in range(steps):
        t = step * dt
        
        # Compute metric norm
        norm = compute_metric_norm(g_xx, g_yy, g_zz)
        norm_history.append(norm)
        time_history.append(t)
        
        # Check for runaway growth (early exit)
        if norm > 100.0 or not np.all(np.isfinite([g_xx, g_yy, g_zz])):
            print(f"  ⚠ Runaway growth or NaN at step {step}, stopping")
            break
        
        # Evolve one timestep
        g_xx, g_yy, g_zz, K_xx, K_yy, K_zz = compute_adm_evolution_step(
            g_xx, g_yy, g_zz, K_xx, K_yy, K_zz,
            alpha, rho_effective, dx, dt, mu_bar, polymer_enabled
        )
    
    # Compute Lyapunov exponent
    lyap = compute_lyapunov_exponent(norm_history, dt)
    
    # Determine stability
    is_stable = lyap < 0.1  # Threshold: small positive growth tolerated
    
    return {
        "parameters": {
            "grid_size": int(grid_size),
            "domain_size": float(domain_size),
            "t_final": float(t_final),
            "dt": float(dt),
            "mu": float(mu),
            "mu_bar": float(mu_bar),
            "R_bubble": float(R_bubble),
            "polymer_enabled": bool(polymer_enabled),
            "synergy_factor": float(synergy_factor),
            "synergy_enabled": bool(abs(synergy_factor) > 1e-12),
        },
        "evolution": {
            "steps_completed": len(norm_history),
            "time_history": time_history,
            "norm_history": norm_history,
        },
        "stability": {
            "lyapunov_exponent": float(lyap),
            "is_stable": bool(is_stable),
            "final_norm": float(norm_history[-1]) if norm_history else 0.0,
            "initial_norm": float(norm_history[0]) if norm_history else 0.0,
            "growth_factor": float(norm_history[-1] / norm_history[0]) if norm_history and norm_history[0] > 0 else 1.0,
        },
        "energy_density": {
            "base_rho": rho_3d,
            "synergistic_rho": rho_syn_3d,
            "synergy_boost": float(1.0 + synergy_factor),
        },
        "final_state": {
            "g_xx": g_xx,
            "g_yy": g_yy,
            "g_zz": g_zz,
            "K_xx": K_xx,
            "K_yy": K_yy,
            "K_zz": K_zz,
        },
        "interpretation": (
            f"{'Stable' if is_stable else 'Unstable'} evolution "
            f"(λ = {lyap:.4f}, growth {norm_history[-1]/norm_history[0]:.2f}× over {time_history[-1]:.3f}s)"
            f"{' [synergy S='+str(synergy_factor)+']' if abs(synergy_factor) > 1e-12 else ' [baseline]'}"
            if norm_history else "Evolution failed"
        ),
    }


def plot_evolution_diagnostics(result: Dict[str, Any], save_path: Path | None = None) -> None:
    """
    Generate diagnostic plots: metric norm evolution, Lyapunov estimate.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    time = result["evolution"]["time_history"]
    norm = result["evolution"]["norm_history"]
    lyap = result["stability"]["lyapunov_exponent"]
    
    # Panel 1: Metric norm evolution
    ax = axes[0]
    ax.plot(time, norm, linewidth=2, label="||g(t)||")
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, label="Flat space")
    ax.set_xlabel("Time")
    ax.set_ylabel("Metric Norm")
    ax.set_title(f"3+1D Metric Evolution (Lyapunov λ = {lyap:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Log-scale growth (for Lyapunov visualization)
    ax = axes[1]
    if len(norm) > 1 and norm[0] > 0:
        log_norm = [np.log(n / norm[0]) for n in norm]
        ax.plot(time, log_norm, linewidth=2, color='red', label="log(||g(t)|| / ||g(0)||)")
        
        # Fit line for Lyapunov exponent
        if len(time) > 2:
            lyap_fit_line = [lyap * t for t in time]
            ax.plot(time, lyap_fit_line, '--', linewidth=1.5, color='orange', 
                   label=f"λt (fit: λ={lyap:.4f})")
        
        ax.set_xlabel("Time")
        ax.set_ylabel("log(Norm Ratio)")
        ax.set_title("Lyapunov Exponent Estimation")
        ax.legend()
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
        description="3+1D stability analysis with polymer-corrected ADM evolution"
    )
    parser.add_argument("--grid-size", type=int, default=32, help="Grid points per dimension (N×N×N)")
    parser.add_argument("--domain-size", type=float, default=5.0, help="Domain half-width")
    parser.add_argument("--t-final", type=float, default=1.0, help="Final evolution time")
    parser.add_argument("--dt", type=float, default=0.001, help="Timestep")
    parser.add_argument("--mu", type=float, default=0.1, help="Polymer parameter μ (energy density)")
    parser.add_argument("--mu-bar", type=float, default=0.1, help="Polymer parameter μ̄ (K correction)")
    parser.add_argument("--R", type=float, default=2.3, help="Bubble radius")
    parser.add_argument("--no-polymer", action="store_true", help="Disable polymer corrections (classical)")
    parser.add_argument("--synergy-factor", type=float, default=0.0, help="Synergy factor S (0 = baseline, no synergy)")
    parser.add_argument("--save-results", action="store_true", help="Save JSON results")
    parser.add_argument("--save-plots", action="store_true", help="Save plots")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("  3+1D Stability Analysis — Polymer-Corrected ADM Evolution")
    print("="*70)
    print(f"\nGrid: {args.grid_size}³ = {args.grid_size**3:,} points")
    print(f"Domain: [-{args.domain_size}, +{args.domain_size}]³")
    print(f"Evolution time: 0 → {args.t_final} (dt = {args.dt})")
    print(f"Polymer: {'ENABLED' if not args.no_polymer else 'DISABLED'} (μ={args.mu}, μ̄={args.mu_bar})")
    if abs(args.synergy_factor) > 1e-12:
        print(f"Synergy: S = {args.synergy_factor:.4f} (boost: {1.0+args.synergy_factor:.3f}×)")
    else:
        print(f"Synergy: BASELINE (S=0, no cross-pathway coupling)")
    print(f"\nRunning evolution...")
    
    # Run evolution
    result = evolve_3d_metric(
        grid_size=args.grid_size,
        domain_size=args.domain_size,
        t_final=args.t_final,
        dt=args.dt,
        mu=args.mu,
        mu_bar=args.mu_bar,
        R_bubble=args.R,
        polymer_enabled=not args.no_polymer,
        synergy_factor=args.synergy_factor,
    )
    
    # Add metadata
    result["timestamp"] = _timestamp()
    result["command_args"] = vars(args)
    
    # Print summary
    print("\n" + "-"*70)
    print("  Results")
    print("-"*70)
    print(f"Steps completed: {result['evolution']['steps_completed']}")
    print(f"Lyapunov exponent λ: {result['stability']['lyapunov_exponent']:.6f}")
    print(f"Stability: {'✓ STABLE' if result['stability']['is_stable'] else '✗ UNSTABLE'}")
    print(f"Growth factor: {result['stability']['growth_factor']:.3f}×")
    print(f"\n{result['interpretation']}")
    print("="*70 + "\n")
    
    # Save results
    if args.save_results:
        # Remove large arrays before saving (keep only summary stats)
        result_to_save = result.copy()
        result_to_save["final_state"] = _json_safe(result["final_state"])
        result_to_save["energy_density"] = {
            "base_rho": _json_safe(result["energy_density"]["base_rho"]),
            "synergistic_rho": _json_safe(result["energy_density"]["synergistic_rho"]),
            "synergy_boost": result["energy_density"]["synergy_boost"],
        }
        
        out_file = results_dir / f"full_3d_evolution_{result['timestamp']}.json"
        out_file.write_text(json.dumps(result_to_save, indent=2), encoding="utf-8")
        print(f"Saved: {out_file}")
    
    # Save plots
    if args.save_plots:
        plot_file = results_dir / f"full_3d_evolution_{result['timestamp']}.png"
        plot_evolution_diagnostics(result, plot_file)
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
