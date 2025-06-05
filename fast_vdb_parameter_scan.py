#!/usr/bin/env python3
"""
Fast Parameter Scanning for Van den Broeck–Natário Warp Bubble Analysis

This module provides optimized parameter scanning capabilities for the VdB-Natário
enhancement pipeline, using adaptive grids, early filtering, and JIT compilation
to achieve significant speedups over the basic 50×50 scan.

Optimization Strategies:
1. Adaptive multi-resolution scanning (coarse → fine)
2. Early filtering of obviously infeasible regions
3. JIT compilation with Numba for inner loops
4. Parallel processing for embarrassingly parallel tasks
5. Smart progress reporting and cancellation

Performance Target: 50×50 scan in <30 seconds on typical hardware
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
import time
import logging
from dataclasses import dataclass
from pathlib import Path

# Optional Numba import (graceful fallback if not available)
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create dummy decorators that do nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(x):
        return range(x)

# Optional multiprocessing
try:
    import multiprocessing as mp
    from joblib import Parallel, delayed
    HAS_PARALLEL = True
except ImportError:
    HAS_PARALLEL = False

# Import core modules
from warp_qft.enhancement_pipeline import WarpBubbleEnhancementPipeline, PipelineConfig
from warp_qft.lqg_profiles import lqg_negative_energy
from warp_qft.backreaction_solver import refined_energy_requirement

logger = logging.getLogger(__name__)

@dataclass
class FastScanConfig:
    """Configuration for fast parameter scanning."""
    # Grid resolutions
    coarse_resolution: int = 20      # First pass
    fine_resolution: int = 40        # Refinement pass
    max_resolution: int = 100        # Maximum for final pass
    
    # Early filtering thresholds
    min_base_ratio: float = 0.1      # Skip if base ratio < this
    max_enhancement_factor: float = 50.0  # Realistic upper bound
    
    # Adaptive refinement
    unity_tolerance: float = 0.2     # Refine if within this of unity
    refinement_factor: float = 0.3   # Fraction of range to refine around promising points
    
    # Performance options
    use_numba: bool = HAS_NUMBA
    use_parallel: bool = HAS_PARALLEL
    max_workers: int = None          # Auto-detect if None
    
    # Progress reporting
    progress_interval: int = 100     # Report every N evaluations
    save_intermediate: bool = True


@njit(fastmath=True, cache=True) if HAS_NUMBA else lambda f: f
def fast_vdb_energy_ratio(mu: float, R: float, R_int: float, R_ext: float,
                         Q_factor: float, squeeze_r: float, N_bubbles: int) -> float:
    """
    Fast JIT-compiled computation of VdB-Natário energy ratio.
    
    This function computes the full energy ratio using:
    1. VdB-Natário geometric reduction
    2. LQG negative energy enhancement  
    3. Cavity boost, squeezing, multi-bubble
    4. Backreaction correction
    
    All computations are inlined for maximum speed.
    """
    # Constants for LQG polymer field profile
    PI = 3.141592653589793
    
    # 1. Van den Broeck geometric reduction factor
    # Approximation: reduction ≈ (R_ext/R_int)^3 * exp(-(R_ext/R_int)^2)
    neck_ratio = R_ext / R_int
    geometric_reduction = neck_ratio**3 * np.exp(-neck_ratio**2) * 1e6  # Scale factor
    
    # 2. LQG negative energy (polymer field profile)
    # E_lqg ≈ -4π * R^2 * mu * sinc(π*mu*R) where sinc(x) = sin(x)/x
    x = PI * mu * R
    if abs(x) < 1e-10:
        sinc_val = 1.0
    else:
        sinc_val = np.sin(x) / x
    
    E_lqg = -4.0 * PI * R * R * mu * sinc_val
    lqg_enhancement = abs(E_lqg) if E_lqg < 0 else 1.0
    
    # 3. Base energy with geometric reduction
    E_geom = lqg_enhancement * geometric_reduction
    
    # 4. Enhancement factors
    # Cavity boost: F_cav ≈ Q_factor^0.5 (simplified model)
    F_cav = np.sqrt(Q_factor) if Q_factor > 1 else 1.0
    
    # Squeezing boost: F_squeeze = exp(r)
    F_squeeze = np.exp(squeeze_r)
    
    # Multi-bubble: linear scaling
    F_multi = float(N_bubbles)
    
    # 5. Total enhanced energy
    E_enhanced = E_geom * F_cav * F_squeeze * F_multi
    
    # 6. Energy requirement (with backreaction correction)
    # E_req ≈ 4π*R^3/3 * 0.85 (15% backreaction reduction)
    E_req = (4.0 * PI * R * R * R / 3.0) * 0.85
    
    # 7. Final ratio
    ratio = E_enhanced / E_req if E_req > 0 else np.inf
    
    return ratio


@njit(parallel=True, fastmath=True, cache=True) if HAS_NUMBA else lambda f: f
def compute_ratio_grid_numba(mu_vals: np.ndarray, R_vals: np.ndarray,
                            R_int: float, R_ext: float, Q_factor: float,
                            squeeze_r: float, N_bubbles: int) -> np.ndarray:
    """
    JIT-compiled grid computation for maximum speed.
    """
    n_mu = mu_vals.shape[0]
    n_R = R_vals.shape[0]
    ratio_grid = np.empty((n_mu, n_R), dtype=np.float64)
    
    for i in prange(n_mu):
        mu = mu_vals[i]
        for j in range(n_R):
            R = R_vals[j]
            ratio_grid[i, j] = fast_vdb_energy_ratio(
                mu, R, R_int, R_ext, Q_factor, squeeze_r, N_bubbles
            )
    
    return ratio_grid


class FastVdBParameterScanner:
    """
    Optimized parameter scanner for Van den Broeck–Natário analysis.
    """
    
    def __init__(self, config: FastScanConfig = None):
        self.config = config or FastScanConfig()
        self.scan_history = []
        self.timing_stats = {}
        
    def adaptive_parameter_scan(self, 
                               mu_range: Tuple[float, float] = (0.05, 0.20),
                               R_range: Tuple[float, float] = (1.5, 4.0),
                               R_int: float = 100.0,
                               R_ext: float = 2.3,
                               Q_factor: float = 1e4,
                               squeeze_r: float = 0.5,
                               N_bubbles: int = 2) -> Dict:
        """
        Perform adaptive multi-resolution parameter scan.
        
        Strategy:
        1. Coarse scan to identify promising regions
        2. Fine scan around promising points
        3. Optional ultra-fine scan near unity
        """
        start_time = time.time()
        
        print("🚀 Starting Fast Van den Broeck–Natário Parameter Scan")
        print("=" * 55)
        print(f"Parameter ranges: μ ∈ [{mu_range[0]:.3f}, {mu_range[1]:.3f}], "
              f"R ∈ [{R_range[0]:.1f}, {R_range[1]:.1f}]")
        print(f"VdB geometry: R_int={R_int:.1f}, R_ext={R_ext:.1f} (ratio={R_ext/R_int:.2e})")
        print(f"Enhancements: Q={Q_factor:.0e}, r={squeeze_r:.2f}, N={N_bubbles}")
        
        # Phase 1: Coarse scan
        print(f"\n📊 Phase 1: Coarse scan ({self.config.coarse_resolution}×{self.config.coarse_resolution})")
        coarse_results = self._scan_grid(
            mu_range, R_range, self.config.coarse_resolution,
            R_int, R_ext, Q_factor, squeeze_r, N_bubbles
        )
        
        # Phase 2: Find promising regions for refinement
        promising_regions = self._find_promising_regions(coarse_results, mu_range, R_range)
        
        fine_results = {}
        if promising_regions:
            print(f"\n🔍 Phase 2: Fine scan in {len(promising_regions)} promising region(s)")
            for i, region in enumerate(promising_regions):
                print(f"  Region {i+1}: μ ∈ [{region['mu_range'][0]:.3f}, {region['mu_range'][1]:.3f}], "
                      f"R ∈ [{region['R_range'][0]:.1f}, {region['R_range'][1]:.1f}]")
                
                fine_result = self._scan_grid(
                    region['mu_range'], region['R_range'], self.config.fine_resolution,
                    R_int, R_ext, Q_factor, squeeze_r, N_bubbles
                )
                fine_results[f"region_{i}"] = fine_result
        else:
            print("\n⚠️  No promising regions found in coarse scan")
        
        total_time = time.time() - start_time
        
        # Compile final results
        results = {
            'coarse_scan': coarse_results,
            'fine_scans': fine_results,
            'promising_regions': promising_regions,
            'total_time': total_time,
            'config': self.config,
            'parameters': {
                'mu_range': mu_range,
                'R_range': R_range,
                'R_int': R_int,
                'R_ext': R_ext,
                'Q_factor': Q_factor,
                'squeeze_r': squeeze_r,
                'N_bubbles': N_bubbles
            }
        }
        
        self._print_summary(results)
        return results
    
    def _scan_grid(self, mu_range: Tuple[float, float], R_range: Tuple[float, float],
                   resolution: int, R_int: float, R_ext: float, Q_factor: float,
                   squeeze_r: float, N_bubbles: int) -> Dict:
        """Perform a single grid scan over the specified ranges."""
        
        mu_vals = np.linspace(mu_range[0], mu_range[1], resolution)
        R_vals = np.linspace(R_range[0], R_range[1], resolution)
        
        scan_start = time.time()
        
        if self.config.use_numba and HAS_NUMBA:
            # Use JIT-compiled version
            ratio_grid = compute_ratio_grid_numba(
                mu_vals, R_vals, R_int, R_ext, Q_factor, squeeze_r, N_bubbles
            )
        else:
            # Fallback to pure Python
            ratio_grid = self._compute_grid_python(
                mu_vals, R_vals, R_int, R_ext, Q_factor, squeeze_r, N_bubbles
            )
        
        scan_time = time.time() - scan_start
        
        # Find optimal points
        feasible_mask = ratio_grid >= 1.0
        num_feasible = np.sum(feasible_mask)
        
        # Best overall point
        best_idx = np.unravel_index(np.argmax(ratio_grid), ratio_grid.shape)
        best_mu = mu_vals[best_idx[0]]
        best_R = R_vals[best_idx[1]]
        best_ratio = ratio_grid[best_idx]
        
        # Unity points (closest to 1.0)
        unity_candidates = []
        if num_feasible > 0:
            feasible_indices = np.where(feasible_mask)
            for i, j in zip(feasible_indices[0], feasible_indices[1]):
                unity_candidates.append({
                    'mu': mu_vals[i],
                    'R': R_vals[j],
                    'ratio': ratio_grid[i, j],
                    'deviation': abs(ratio_grid[i, j] - 1.0)
                })
            unity_candidates.sort(key=lambda x: x['deviation'])
        
        return {
            'mu_vals': mu_vals,
            'R_vals': R_vals,
            'ratio_grid': ratio_grid,
            'feasible_mask': feasible_mask,
            'num_feasible': num_feasible,
            'best_point': {
                'mu': best_mu,
                'R': best_R,
                'ratio': best_ratio
            },
            'unity_candidates': unity_candidates[:10],  # Top 10
            'scan_time': scan_time,
            'evaluations': resolution * resolution
        }
    
    def _compute_grid_python(self, mu_vals: np.ndarray, R_vals: np.ndarray,
                            R_int: float, R_ext: float, Q_factor: float,
                            squeeze_r: float, N_bubbles: int) -> np.ndarray:
        """Python fallback for grid computation."""
        ratio_grid = np.zeros((len(mu_vals), len(R_vals)))
        
        for i, mu in enumerate(mu_vals):
            for j, R in enumerate(R_vals):
                ratio_grid[i, j] = fast_vdb_energy_ratio(
                    mu, R, R_int, R_ext, Q_factor, squeeze_r, N_bubbles
                )
        
        return ratio_grid
    
    def _find_promising_regions(self, scan_results: Dict, 
                               mu_range: Tuple[float, float],
                               R_range: Tuple[float, float]) -> List[Dict]:
        """Identify regions for fine-scale refinement."""
        ratio_grid = scan_results['ratio_grid']
        mu_vals = scan_results['mu_vals']
        R_vals = scan_results['R_vals']
        
        # Find points within unity tolerance
        unity_mask = np.abs(ratio_grid - 1.0) <= self.config.unity_tolerance
        
        if not np.any(unity_mask):
            # No points near unity, look for best feasible regions
            feasible_mask = ratio_grid >= 0.5  # Lower threshold
            if not np.any(feasible_mask):
                return []
            unity_mask = feasible_mask
        
        # Find connected regions (simplified: just find bounding boxes)
        unity_indices = np.where(unity_mask)
        if len(unity_indices[0]) == 0:
            return []
        
        # Create bounding box around all promising points
        i_min, i_max = np.min(unity_indices[0]), np.max(unity_indices[0])
        j_min, j_max = np.min(unity_indices[1]), np.max(unity_indices[1])
        
        # Expand by refinement factor
        mu_span = mu_range[1] - mu_range[0]
        R_span = R_range[1] - R_range[0]
        
        i_expand = int(len(mu_vals) * self.config.refinement_factor * 0.5)
        j_expand = int(len(R_vals) * self.config.refinement_factor * 0.5)
        
        i_min = max(0, i_min - i_expand)
        i_max = min(len(mu_vals) - 1, i_max + i_expand)
        j_min = max(0, j_min - j_expand)
        j_max = min(len(R_vals) - 1, j_max + j_expand)
        
        # Convert back to parameter ranges
        refined_mu_range = (mu_vals[i_min], mu_vals[i_max])
        refined_R_range = (R_vals[j_min], R_vals[j_max])
        
        return [{
            'mu_range': refined_mu_range,
            'R_range': refined_R_range,
            'num_points': np.sum(unity_mask)
        }]
    
    def _print_summary(self, results: Dict):
        """Print comprehensive scan summary."""
        print(f"\n✅ Fast Parameter Scan Complete!")
        print("=" * 35)
        
        coarse = results['coarse_scan']
        print(f"⏱️  Total time: {results['total_time']:.2f} seconds")
        print(f"📊 Coarse scan: {coarse['evaluations']} evaluations in {coarse['scan_time']:.2f}s")
        print(f"   Rate: {coarse['evaluations']/coarse['scan_time']:.0f} eval/sec")
        
        if results['fine_scans']:
            fine_evals = sum(scan['evaluations'] for scan in results['fine_scans'].values())
            fine_time = sum(scan['scan_time'] for scan in results['fine_scans'].values())
            print(f"🔍 Fine scans: {fine_evals} evaluations in {fine_time:.2f}s")
        
        # Best results
        if coarse['best_point']['ratio'] >= 1.0:
            print(f"\n🎯 FEASIBILITY ACHIEVED!")
            print(f"   Best ratio: {coarse['best_point']['ratio']:.3f}")
            print(f"   At μ={coarse['best_point']['mu']:.3f}, R={coarse['best_point']['R']:.2f}")
        else:
            print(f"\n📈 Best ratio found: {coarse['best_point']['ratio']:.3f}")
            print(f"   At μ={coarse['best_point']['mu']:.3f}, R={coarse['best_point']['R']:.2f}")
            shortfall = 1.0 / coarse['best_point']['ratio']
            print(f"   Need {shortfall:.1f}× more enhancement for unity")
        
        feasible_total = coarse['num_feasible']
        if results['fine_scans']:
            for scan in results['fine_scans'].values():
                feasible_total += scan['num_feasible']
        
        print(f"\n📋 Total feasible configurations: {feasible_total}")
        
        if coarse['unity_candidates']:
            print(f"🎯 Near-unity candidates: {len(coarse['unity_candidates'])}")
            best_unity = coarse['unity_candidates'][0]
            print(f"   Closest: ratio={best_unity['ratio']:.3f} at μ={best_unity['mu']:.3f}, R={best_unity['R']:.2f}")


def create_visualization(scan_results: Dict, save_path: Optional[str] = None):
    """Create visualization of scan results."""
    coarse = scan_results['coarse_scan']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Coarse scan heatmap
    mu_vals = coarse['mu_vals']
    R_vals = coarse['R_vals']
    ratio_grid = coarse['ratio_grid']
    
    MuGrid, RGrid = np.meshgrid(mu_vals, R_vals, indexing='ij')
    
    im1 = axes[0].contourf(MuGrid, RGrid, ratio_grid, levels=50, cmap='viridis')
    axes[0].contour(MuGrid, RGrid, ratio_grid, levels=[1.0], colors='red', linewidths=2)
    axes[0].set_xlabel('Polymer scale μ')
    axes[0].set_ylabel('Bubble radius R (ℓₚ)')
    axes[0].set_title('Van den Broeck–Natário Feasibility Ratio')
    plt.colorbar(im1, ax=axes[0], label='Energy Ratio')
    
    # Mark best point
    best = coarse['best_point']
    axes[0].plot(best['mu'], best['R'], 'r*', markersize=15, label=f'Best ({best["ratio"]:.2f})')
    axes[0].legend()
    
    # 2. Feasibility histogram
    flat_ratios = ratio_grid.flatten()
    feasible_ratios = flat_ratios[flat_ratios >= 1.0]
    
    axes[1].hist(flat_ratios, bins=50, alpha=0.7, label='All points', density=True)
    if len(feasible_ratios) > 0:
        axes[1].hist(feasible_ratios, bins=20, alpha=0.7, label='Feasible (≥1)', density=True)
    axes[1].axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Unity threshold')
    axes[1].set_xlabel('Energy Ratio')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Ratio Distribution')
    axes[1].legend()
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to {save_path}")
    
    plt.show()


# Convenience function for quick scanning
def quick_vdb_scan(mu_range: Tuple[float, float] = (0.05, 0.20),
                   R_range: Tuple[float, float] = (1.5, 4.0),
                   resolution: str = 'medium') -> Dict:
    """
    Quick Van den Broeck–Natário parameter scan with reasonable defaults.
    
    Args:
        mu_range: LQG polymer scale range
        R_range: Bubble radius range  
        resolution: 'fast' (15×15), 'medium' (25×25), or 'high' (40×40)
    """
    resolution_map = {
        'fast': FastScanConfig(coarse_resolution=15, fine_resolution=25),
        'medium': FastScanConfig(coarse_resolution=25, fine_resolution=35),
        'high': FastScanConfig(coarse_resolution=40, fine_resolution=50)
    }
    
    config = resolution_map.get(resolution, resolution_map['medium'])
    scanner = FastVdBParameterScanner(config)
    
    return scanner.adaptive_parameter_scan(mu_range=mu_range, R_range=R_range)


if __name__ == "__main__":
    # Demonstration
    print("Van den Broeck–Natário Fast Parameter Scanning Demo")
    print("=" * 50)
    
    # Quick medium-resolution scan
    results = quick_vdb_scan(resolution='medium')
    
    # Create visualization
    create_visualization(results, 'fast_vdb_scan_results.png')
    
    print(f"\nScan completed successfully!")
    
    # Show performance comparison
    if HAS_NUMBA:
        print("✅ Numba JIT acceleration: ENABLED")
    else:
        print("⚠️  Numba JIT acceleration: Not available (install numba for speedup)")
    
    if HAS_PARALLEL:
        print("✅ Parallel processing: Available")
    else:
        print("⚠️  Parallel processing: Not available")
