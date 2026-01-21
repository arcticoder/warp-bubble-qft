#!/usr/bin/env python3
"""
Sensitivity & Robustness Analysis for Warp Bubble Feasibility

This script performs Monte Carlo sampling and sensitivity analysis to test
whether the claimed feasibility is robust to:
1. Parameter uncertainties (noise in μ, R)
2. Variation in enhancement factors (cavity Q, squeezing, multi-bubble)
3. Different initial conditions

Usage:
    python sensitivity_analysis.py [--trials N] [--save-results]
"""

import argparse
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent / "src"))

from src.warp_qft.enhancement_pipeline import (
    WarpBubbleEnhancementPipeline, PipelineConfig
)
from src.warp_qft.enhancement_pathway import EnhancementConfig


def monte_carlo_feasibility_test(
    n_trials: int = 100,
    mu_range: tuple = (0.05, 0.20),
    R_range: tuple = (1.5, 4.0),
    noise_level: float = 0.05,
) -> dict:
    """
    Run Monte Carlo sampling to test feasibility robustness.
    
    Args:
        n_trials: Number of random trials
        mu_range: (min, max) for μ sampling
        R_range: (min, max) for R sampling
        noise_level: Relative noise amplitude (e.g., 0.05 = 5%)
    
    Returns:
        dict with trial results and statistics
    """
    config = PipelineConfig()
    pipeline = WarpBubbleEnhancementPipeline(config)
    
    results = []
    feasible_count = 0
    
    np.random.seed(42)  # Reproducible randomness
    
    print(f"Running {n_trials} Monte Carlo trials...")
    for i in range(n_trials):
        # Sample parameters uniformly
        mu = np.random.uniform(mu_range[0], mu_range[1])
        R = np.random.uniform(R_range[0], R_range[1])
        
        # Add Gaussian noise
        mu_noisy = mu * (1 + noise_level * np.random.randn())
        R_noisy = R * (1 + noise_level * np.random.randn())
        
        # Clamp to physical ranges
        mu_noisy = np.clip(mu_noisy, 0.01, 1.0)
        R_noisy = np.clip(R_noisy, 0.1, 10.0)
        
        # Compute energy
        try:
            base_energy = pipeline.compute_base_energy_requirement(mu_noisy, R_noisy)
            corrections = pipeline.apply_all_corrections(base_energy, mu_noisy, R_noisy)
            final_energy = corrections["final_energy"]
            
            is_feasible = final_energy <= 1.0
            if is_feasible:
                feasible_count += 1
            
            results.append({
                "trial": i,
                "mu": float(mu_noisy),
                "R": float(R_noisy),
                "base_energy": float(base_energy),
                "final_energy": float(final_energy),
                "feasible": bool(is_feasible),
            })
        except Exception as e:
            print(f"  Trial {i} failed: {e}")
            results.append({
                "trial": i,
                "mu": mu_noisy,
                "R": R_noisy,
                "error": str(e),
            })
    
    feasibility_rate = feasible_count / n_trials
    energies = [r["final_energy"] for r in results if "final_energy" in r]
    
    return {
        "n_trials": n_trials,
        "feasible_count": feasible_count,
        "feasibility_rate": feasibility_rate,
        "noise_level": noise_level,
        "mu_range": mu_range,
        "R_range": R_range,
        "results": results,
        "energy_stats": {
            "mean": float(np.mean(energies)),
            "median": float(np.median(energies)),
            "std": float(np.std(energies)),
            "min": float(np.min(energies)),
            "max": float(np.max(energies)),
            "percentile_5": float(np.percentile(energies, 5)),
            "percentile_95": float(np.percentile(energies, 95)),
        },
    }


def enhancement_factor_sensitivity(
    mu: float = 0.1,
    R: float = 2.3,
    vary_factor: str = "cavity_Q",
    factor_range: tuple = (1e4, 1e8),
    n_points: int = 20,
) -> dict:
    """
    Test sensitivity to individual enhancement factors.
    
    Args:
        mu: Fixed polymer scale
        R: Fixed bubble radius
        vary_factor: Which factor to vary ('cavity_Q', 'squeezing_db', 'num_bubbles')
        factor_range: (min, max) for the varied factor
        n_points: Number of sample points
    
    Returns:
        dict with sensitivity results
    """
    config = PipelineConfig()
    pipeline = WarpBubbleEnhancementPipeline(config)
    
    base_energy = pipeline.compute_base_energy_requirement(mu, R)
    
    if vary_factor == "cavity_Q":
        values = np.logspace(np.log10(factor_range[0]), np.log10(factor_range[1]), n_points)
    elif vary_factor == "squeezing_db":
        values = np.linspace(factor_range[0], factor_range[1], n_points)
    elif vary_factor == "num_bubbles":
        values = np.arange(int(factor_range[0]), int(factor_range[1]) + 1)
    else:
        raise ValueError(f"Unknown factor: {vary_factor}")
    
    results = []
    
    for val in values:
        # Update config
        if vary_factor == "cavity_Q":
            config.enhancement_config.cavity_Q = val
        elif vary_factor == "squeezing_db":
            config.enhancement_config.squeezing_db = val
        elif vary_factor == "num_bubbles":
            config.enhancement_config.num_bubbles = int(val)
        
        # Rebuild pipeline with updated config
        pipeline_var = WarpBubbleEnhancementPipeline(config)
        corrections = pipeline_var.apply_all_corrections(base_energy, mu, R)
        final_energy = corrections["final_energy"]
        
        results.append({
            "factor_value": float(val),
            "final_energy": float(final_energy),
            "feasible": bool(final_energy <= 1.0),
        })
    
    # Reset config
    config = PipelineConfig()
    
    return {
        "mu": mu,
        "R": R,
        "base_energy": base_energy,
        "varied_factor": vary_factor,
        "results": results,
    }


def plot_monte_carlo_results(mc_result: dict, save_path: str = None):
    """Plot Monte Carlo feasibility distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    energies = [r["final_energy"] for r in mc_result["results"] if "final_energy" in r]
    feasible = [r for r in mc_result["results"] if r.get("feasible", False)]
    infeasible = [r for r in mc_result["results"] if not r.get("feasible", True)]
    
    # Histogram of energies
    ax1.hist(energies, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Unity Threshold')
    ax1.axvline(np.median(energies), color='green', linestyle=':', linewidth=2, label='Median')
    ax1.set_xlabel('Final Energy Requirement', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Monte Carlo Energy Distribution (N={mc_result["n_trials"]})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter: parameter space
    if feasible:
        ax2.scatter([r["mu"] for r in feasible], [r["R"] for r in feasible],
                   c='green', alpha=0.6, s=30, label=f'Feasible ({len(feasible)})')
    if infeasible:
        ax2.scatter([r["mu"] for r in infeasible], [r["R"] for r in infeasible],
                   c='red', alpha=0.6, s=30, label=f'Infeasible ({len(infeasible)})')
    
    ax2.set_xlabel('Polymer Scale μ', fontsize=12)
    ax2.set_ylabel('Bubble Radius R', fontsize=12)
    ax2.set_title('Parameter Space Feasibility', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved Monte Carlo plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_sensitivity_scan(sens_result: dict, save_path: str = None):
    """Plot enhancement factor sensitivity."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    factor_vals = [r["factor_value"] for r in sens_result["results"]]
    energies = [r["final_energy"] for r in sens_result["results"]]
    
    ax.plot(factor_vals, energies, 'b-o', linewidth=2, markersize=6)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Unity Threshold')
    
    if sens_result["varied_factor"] == "cavity_Q":
        ax.set_xscale('log')
        ax.set_xlabel('Cavity Q-Factor', fontsize=12)
    elif sens_result["varied_factor"] == "squeezing_db":
        ax.set_xlabel('Squeezing (dB)', fontsize=12)
    elif sens_result["varied_factor"] == "num_bubbles":
        ax.set_xlabel('Number of Bubbles', fontsize=12)
    
    ax.set_ylabel('Final Energy Requirement', fontsize=12)
    ax.set_title(f'Sensitivity to {sens_result["varied_factor"]}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sensitivity plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Sensitivity and robustness analysis")
    parser.add_argument("--trials", type=int, default=100, help="Number of Monte Carlo trials")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise level (0.05 = 5%)")
    parser.add_argument("--save-results", action="store_true", help="Save results to JSON")
    parser.add_argument("--save-plots", action="store_true", help="Save plots to results/")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SENSITIVITY & ROBUSTNESS ANALYSIS")
    print("=" * 60)
    
    # 1. Monte Carlo feasibility test
    print("\n1. Monte Carlo Feasibility Test")
    print("-" * 60)
    mc_result = monte_carlo_feasibility_test(n_trials=args.trials, noise_level=args.noise)
    print(f"  Trials run: {mc_result['n_trials']}")
    print(f"  Feasible: {mc_result['feasible_count']} ({mc_result['feasibility_rate']*100:.1f}%)")
    print(f"  Energy stats:")
    for key, val in mc_result["energy_stats"].items():
        print(f"    {key}: {val:.6f}")
    
    # 2. Cavity Q sensitivity
    print("\n2. Cavity Q-Factor Sensitivity")
    print("-" * 60)
    cavity_sens = enhancement_factor_sensitivity(vary_factor="cavity_Q", factor_range=(1e4, 1e8))
    feasible_Q = [r for r in cavity_sens["results"] if r["feasible"]]
    print(f"  Feasible Q values: {len(feasible_Q)}/{len(cavity_sens['results'])}")
    
    # 3. Squeezing sensitivity
    print("\n3. Squeezing Parameter Sensitivity")
    print("-" * 60)
    sqz_sens = enhancement_factor_sensitivity(vary_factor="squeezing_db", factor_range=(0, 30))
    feasible_sqz = [r for r in sqz_sens["results"] if r["feasible"]]
    print(f"  Feasible squeezing values: {len(feasible_sqz)}/{len(sqz_sens['results'])}")
    
    # Save results
    if args.save_results:
        Path("results").mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output = {
            "timestamp": timestamp,
            "monte_carlo": mc_result,
            "cavity_Q_sensitivity": cavity_sens,
            "squeezing_sensitivity": sqz_sens,
        }
        
        filename = f"results/sensitivity_analysis_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Results saved to {filename}")
    
    # Save plots
    if args.save_plots:
        Path("results").mkdir(exist_ok=True)
        plot_monte_carlo_results(mc_result, "results/monte_carlo_feasibility.png")
        plot_sensitivity_scan(cavity_sens, "results/cavity_Q_sensitivity.png")
        plot_sensitivity_scan(sqz_sens, "results/squeezing_sensitivity.png")
    
    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
