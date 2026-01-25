#!/usr/bin/env python3
"""Integrated curved QI + 3D stability verification.

Combines curved-spacetime quantum inequality checks with 3+1D metric evolution stability analysis.
Tests whether QI-violating configurations also exhibit dynamical instabilities.

Workflow:
1. Compute curved-space QI integral using Gaussian-sampled energy density
2. Run 3D ADM-like evolution with polymer corrections
3. Cross-check: do QI violations correlate with Lyapunov instabilities?
4. Generate combined diagnostic report + plots

Usage:
    python integrated_qi_3d_verification.py --mu 0.3 --R 2.3 --save-results --save-plots
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_safe(obj: Any) -> Any:
    """Convert obj to JSON-serializable types."""
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


def polymer_energy_density(x: np.ndarray, y: np.ndarray, z: np.ndarray, mu: float, R: float) -> np.ndarray:
    """Compute polymer-modified negative energy density on 3D grid.
    
    Args:
        x, y, z: 3D coordinate grids
        mu: Polymer parameter μ̄
        R: Bubble radius
        
    Returns:
        Negative energy density ρ(x,y,z)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Base Gaussian profile
    rho_base = -np.exp(-(r**2) / (R**2))
    
    # Polymer modification (simplified): sin(μ̄ρ) / (μ̄ρ) ≈ 1 - (μ̄ρ)²/6 for small μ̄ρ
    if mu > 0:
        arg = mu * np.abs(rho_base)
        with np.errstate(divide='ignore', invalid='ignore'):
            polymer_factor = np.where(arg < 1e-8, 1.0 - arg**2 / 6.0, np.sin(arg) / arg)
    else:
        polymer_factor = 1.0
    
    return rho_base * polymer_factor


def curved_qi_integral(
    rho_samples: np.ndarray,
    g_tt: np.ndarray,
    dt: float
) -> float:
    """Compute curved-space QI integral ∫ρ g_tt dτ.
    
    Args:
        rho_samples: Energy density samples ρ(τ_i)
        g_tt: Metric time-time component g_tt(τ_i)
        dt: Proper time spacing
        
    Returns:
        QI integral value
    """
    # Proper time element: dτ² = |g_tt| dt²
    # Integral: ∫ρ(τ) √|g_tt| dτ ≈ Σ ρ_i √|g_tt,i| Δt
    integrand = rho_samples * np.sqrt(np.abs(g_tt))
    return float(np.trapezoid(integrand, dx=dt))


def alcubierre_metric_components(r: np.ndarray, R: float, v_s: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Alcubierre metric components (toy model).
    
    Args:
        r: Radial coordinate array
        R: Bubble radius
        v_s: Effective warp velocity (dimensionless)
        
    Returns:
        (g_tt, g_rr) metric components
    """
    # Shape function
    f = np.tanh((R - r) / (0.1 * R))
    
    # Alcubierre metric (simplified, 1D)
    g_tt = -(1 - v_s**2 * f**2)
    g_rr = 1 + 0.1 * f
    
    return g_tt, g_rr


def run_3d_evolution(
    N: int,
    L: float,
    mu: float,
    R: float,
    t_final: float,
    dt: float,
    polymer_enabled: bool = True
) -> Dict[str, Any]:
    """Run simplified 3D ADM-like metric evolution.
    
    Args:
        N: Grid points per dimension
        L: Domain half-width
        mu: Polymer parameter
        R: Bubble radius
        t_final: Final time
        dt: Time step
        polymer_enabled: Whether to apply polymer corrections
        
    Returns:
        Evolution diagnostics including Lyapunov exponent
    """
    # 3D grid
    x1d = np.linspace(-L, L, N)
    x, y, z = np.meshgrid(x1d, x1d, x1d, indexing='ij')
    
    # Initial metric: flat + small perturbation
    r = np.sqrt(x**2 + y**2 + z**2)
    f_profile = np.tanh((R - r) / (0.1 * R))
    
    g_xx = np.ones_like(x) + 0.01 * f_profile
    g_yy = np.ones_like(x) + 0.01 * f_profile
    g_zz = np.ones_like(x) + 0.01 * f_profile
    
    # Extrinsic curvature (start small)
    K_xx = 0.001 * f_profile
    K_yy = 0.001 * f_profile
    K_zz = 0.001 * f_profile
    
    # Energy density source
    rho = polymer_energy_density(x, y, z, mu, R)
    
    # Time evolution
    n_steps = int(t_final / dt)
    norm_history = []
    
    g_norm_0 = np.sqrt(np.mean(g_xx**2 + g_yy**2 + g_zz**2))
    
    for step in range(n_steps):
        # Polymer correction to extrinsic curvature (if enabled)
        if polymer_enabled and mu > 0:
            for K in [K_xx, K_yy, K_zz]:
                arg = mu * np.abs(K)
                with np.errstate(divide='ignore', invalid='ignore'):
                    K[:] = np.where(arg < 1e-8, K, np.sign(K) * np.sin(arg) / mu)
        
        # Simplified ADM step: ∂_t g_ij = -2α K_ij (lapse α=1)
        g_xx -= 2.0 * dt * K_xx
        g_yy -= 2.0 * dt * K_yy
        g_zz -= 2.0 * dt * K_zz
        
        # Simplified evolution of K_ij (toy model: driven by ρ)
        # ∂_t K_ij ≈ -∇²g_ij + source(ρ)
        # For simplicity: K_ij evolves slowly, decay toward equilibrium
        K_xx *= (1.0 - 0.01 * dt)
        K_yy *= (1.0 - 0.01 * dt)
        K_zz *= (1.0 - 0.01 * dt)
        
        # Add small source term from energy density
        source_strength = 0.001
        K_xx += source_strength * dt * rho
        K_yy += source_strength * dt * rho
        K_zz += source_strength * dt * rho
        
        # Compute norm
        g_norm = np.sqrt(np.mean(g_xx**2 + g_yy**2 + g_zz**2))
        norm_history.append(float(g_norm))
        
        # Safety: detect blowup
        if not np.isfinite(g_norm) or g_norm > 1e6:
            break
    
    # Lyapunov exponent estimate
    if len(norm_history) > 1 and g_norm_0 > 0:
        g_norm_final = norm_history[-1]
        T_actual = len(norm_history) * dt
        lyapunov = np.log(g_norm_final / g_norm_0) / T_actual if T_actual > 0 else 0.0
    else:
        lyapunov = 0.0
    
    return {
        "grid_size": int(N),
        "domain_width": float(L),
        "time_final": float(len(norm_history) * dt),
        "steps": int(len(norm_history)),
        "polymer_enabled": bool(polymer_enabled),
        "norm_history": norm_history,
        "lyapunov_exponent": float(lyapunov),
        "final_norm": float(norm_history[-1]) if norm_history else 0.0,
        "growth_factor": float(norm_history[-1] / g_norm_0) if norm_history and g_norm_0 > 0 else 1.0,
        "stable": bool(lyapunov < 0.01),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mu", type=float, default=0.3, help="Polymer parameter μ̄")
    parser.add_argument("--R", type=float, default=2.3, help="Bubble radius")
    parser.add_argument("--delta-t-qi", type=float, default=1.0, help="Sampling time for QI integral")
    parser.add_argument("--n-samples-qi", type=int, default=100, help="Number of samples for QI integral")
    
    parser.add_argument("--grid-3d", type=int, default=16, help="3D grid size N³")
    parser.add_argument("--domain-width", type=float, default=5.0, help="3D domain half-width")
    parser.add_argument("--t-final-3d", type=float, default=0.5, help="3D evolution final time")
    parser.add_argument("--dt-3d", type=float, default=0.001, help="3D evolution time step")
    
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--save-plots", action="store_true")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Part 1: Curved QI check ---
    print("Computing curved-space QI integral...")
    
    tau_samples = np.linspace(0, args.delta_t_qi, args.n_samples_qi)
    r_samples = np.linspace(0, 3 * args.R, args.n_samples_qi)
    
    # Energy density samples
    x_samples = r_samples
    y_samples = np.zeros_like(r_samples)
    z_samples = np.zeros_like(r_samples)
    
    # Compute on 1D slice (along x-axis)
    r_grid = np.sqrt(x_samples**2)
    rho_base = -np.exp(-(r_grid**2) / (args.R**2))
    
    if args.mu > 0:
        arg = args.mu * np.abs(rho_base)
        with np.errstate(divide='ignore', invalid='ignore'):
            polymer_factor = np.where(arg < 1e-8, 1.0 - arg**2 / 6.0, np.sin(arg) / arg)
    else:
        polymer_factor = 1.0
    
    rho_samples = rho_base * polymer_factor
    
    # Flat-space integral (g_tt = -1)
    qi_flat = float(np.trapezoid(rho_samples, dx=args.delta_t_qi / args.n_samples_qi))
    
    # Curved-space integral (Alcubierre metric)
    g_tt_curved, _ = alcubierre_metric_components(r_samples, args.R)
    qi_curved = curved_qi_integral(rho_samples, g_tt_curved, args.delta_t_qi / args.n_samples_qi)
    
    # Toy bounds (heuristic)
    bound_flat = -1.0 / (16 * np.pi**2 * args.delta_t_qi**2)  # Ford-Roman flat-space
    bound_curved = -1.0 / (args.R**2)  # Toy curved bound (heuristic scaling)
    
    qi_results = {
        "flat_space": {
            "integral": float(qi_flat),
            "bound": float(bound_flat),
            "violates": bool(qi_flat < bound_flat),
            "margin": float(qi_flat - bound_flat),
        },
        "curved_space": {
            "integral": float(qi_curved),
            "bound": float(bound_curved),
            "violates": bool(qi_curved < bound_curved),
            "margin": float(qi_curved - bound_curved),
        },
        "metric_enhancement_factor": float(qi_curved / qi_flat) if qi_flat != 0 else 1.0,
    }
    
    print(f"  Flat QI:   {qi_flat:.6f} vs bound {bound_flat:.6f} → {'VIOLATES' if qi_results['flat_space']['violates'] else 'OK'}")
    print(f"  Curved QI: {qi_curved:.6f} vs bound {bound_curved:.6f} → {'VIOLATES' if qi_results['curved_space']['violates'] else 'OK'}")
    
    # --- Part 2: 3D evolution stability ---
    print("\nRunning 3D metric evolution...")
    
    evolution_results = run_3d_evolution(
        args.grid_3d,
        args.domain_width,
        args.mu,
        args.R,
        args.t_final_3d,
        args.dt_3d,
        polymer_enabled=True
    )
    
    print(f"  Lyapunov λ = {evolution_results['lyapunov_exponent']:.6f}, growth = {evolution_results['growth_factor']:.4f}×")
    print(f"  Stability: {'STABLE' if evolution_results['stable'] else 'UNSTABLE'}")
    
    # --- Part 3: Correlation analysis ---
    print("\nIntegrated analysis...")
    
    # Hypothesis: QI violation should correlate with instability
    qi_violation_score = max(0, -(qi_results['curved_space']['margin']))  # More negative margin → higher violation
    instability_score = max(0, evolution_results['lyapunov_exponent'])  # Positive λ → unstable
    
    correlation_analysis = {
        "qi_violation_score": float(qi_violation_score),
        "instability_score": float(instability_score),
        "both_safe": bool(not qi_results['curved_space']['violates'] and evolution_results['stable']),
        "both_problematic": bool(qi_results['curved_space']['violates'] and not evolution_results['stable']),
        "interpretation": (
            "No violations detected; configuration appears consistent."
            if not qi_results['curved_space']['violates'] and evolution_results['stable']
            else "QI violates but evolution stable; may indicate toy-model limitations."
            if qi_results['curved_space']['violates'] and evolution_results['stable']
            else "Evolution unstable; check QI violation for energy condition breach."
        ),
    }
    
    print(f"  QI violation score: {qi_violation_score:.6f}")
    print(f"  Instability score:  {instability_score:.6f}")
    print(f"  → {correlation_analysis['interpretation']}")
    
    # --- Assemble report ---
    report = {
        "timestamp": _timestamp(),
        "parameters": {
            "mu": float(args.mu),
            "R": float(args.R),
            "delta_t_qi": float(args.delta_t_qi),
            "n_samples_qi": int(args.n_samples_qi),
            "grid_3d": int(args.grid_3d),
            "t_final_3d": float(args.t_final_3d),
        },
        "curved_qi_analysis": qi_results,
        "3d_evolution_analysis": evolution_results,
        "correlation_analysis": correlation_analysis,
    }
    
    if args.save_results:
        out_path = results_dir / f"integrated_qi_3d_{report['timestamp']}.json"
        out_path.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
        print(f"\nWrote {out_path}")
    
    if args.save_plots:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Plot 1: QI integral comparison
            labels = ['Flat\n(Ford-Roman)', 'Curved\n(Alcubierre)']
            integrals = [qi_results['flat_space']['integral'], qi_results['curved_space']['integral']]
            bounds = [qi_results['flat_space']['bound'], qi_results['curved_space']['bound']]
            
            x_pos = np.arange(len(labels))
            ax1.bar(x_pos - 0.2, integrals, 0.4, label='QI Integral', alpha=0.7)
            ax1.bar(x_pos + 0.2, bounds, 0.4, label='Bound', alpha=0.7)
            ax1.axhline(0, color='k', linestyle='--', linewidth=0.5)
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(labels)
            ax1.set_ylabel('Energy integral')
            ax1.set_title('Quantum Inequality Check')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: 3D evolution stability
            times = np.arange(len(evolution_results['norm_history'])) * args.dt_3d
            ax2.plot(times, evolution_results['norm_history'], label='Metric norm ||g||')
            ax2.axhline(evolution_results['norm_history'][0] if evolution_results['norm_history'] else 1.0, 
                       color='gray', linestyle='--', alpha=0.5, label='Initial')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('||g||')
            ax2.set_title(f'3D Evolution (λ={evolution_results["lyapunov_exponent"]:.2e})')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = results_dir / f"integrated_qi_3d_{report['timestamp']}.png"
            plt.savefig(plot_path, dpi=160)
            plt.close()
            print(f"Wrote {plot_path}")
        except Exception as exc:
            print(f"Plotting failed: {exc}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
