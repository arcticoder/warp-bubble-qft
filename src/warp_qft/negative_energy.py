"""
Negative Energy and Warp Bubble Formation

Analysis of stable negative energy densities and warp bubble configurations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .field_algebra import PolymerField
import logging

logger = logging.getLogger(__name__)


class WarpBubble:
    """
    Represents a stable negative energy configuration (warp bubble).
    """
    
    def __init__(self, center_position: float, bubble_radius: float, 
                 negative_energy_density: float, polymer_scale: float):
        """
        Initialize warp bubble configuration.
        
        Args:
            center_position: Center of the bubble (0 to 1)
            bubble_radius: Spatial extent of negative energy region
            negative_energy_density: Peak negative energy density
            polymer_scale: Polymer parameter μ̄
        """
        self.center = center_position
        self.radius = bubble_radius
        self.rho_neg = negative_energy_density
        self.mu_bar = polymer_scale
        
        # Stability parameters
        self.formation_time = 0.0
        self.decay_time = np.inf
        self.is_stable = True
    
    def energy_profile(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the energy density profile of the warp bubble.
        
        Uses a modified Gaussian profile with polymer corrections.
        """
        # Distance from bubble center
        r = np.abs(x - self.center)
        
        # Classical Gaussian profile
        classical_profile = self.rho_neg * np.exp(-(r/self.radius)**2)
        
        # Polymer corrections (oscillatory modifications)
        if self.mu_bar > 0:
            polymer_correction = 1 + 0.1 * self.mu_bar * np.sin(2*np.pi*r/self.mu_bar)
            return classical_profile * polymer_correction
        else:
            return classical_profile
    
    def compute_total_negative_energy(self, x_grid: np.ndarray) -> float:
        """Compute total negative energy in the bubble."""
        energy_density = self.energy_profile(x_grid)
        negative_regions = energy_density < 0
        if not np.any(negative_regions):
            return 0.0
        
        dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
        return np.sum(energy_density[negative_regions]) * dx
    
    def stability_analysis(self, duration: float = 10.0) -> Dict:
        """
        Analyze the stability of the warp bubble over time.
        
        Returns:
            Dictionary with stability metrics
        """
        # Simplified stability analysis
        # In reality, this would involve solving the full field equations
        
        # Classical instability timescale (Ford-Roman bound)
        classical_lifetime = self.radius / (np.abs(self.rho_neg) + 1e-10)
        
        # Polymer stabilization factor
        if self.mu_bar > 0:
            stabilization_factor = 1 + self.mu_bar**2
            polymer_lifetime = classical_lifetime * stabilization_factor
        else:
            polymer_lifetime = classical_lifetime
        
        # Check if bubble survives the specified duration
        survives_duration = polymer_lifetime > duration
        
        return {
            "classical_lifetime": classical_lifetime,
            "polymer_lifetime": polymer_lifetime,
            "stabilization_factor": polymer_lifetime / classical_lifetime,
            "survives_duration": survives_duration,
            "stability_ratio": polymer_lifetime / duration
        }


def compute_negative_energy_region(lattice_size: int, polymer_scale: float,
                                 field_amplitude: float = 1.0) -> Dict:
    """
    Compute negative energy regions in a polymer field configuration.
    
    Args:
        lattice_size: Number of lattice sites
        polymer_scale: Polymer parameter μ̄
        field_amplitude: Initial field amplitude
        
    Returns:
        Dictionary with negative energy analysis
    """
    # Create polymer field
    field = PolymerField(lattice_size, polymer_scale)
    
    # Set up initial configuration for negative energy formation
    # Use a specific coherent state that promotes negative energy
    width = 0.1
    field.set_coherent_state(field_amplitude, width, center=0.5)
    
    # Add momentum to create interference patterns
    x = np.linspace(0, 1, lattice_size)
    field.pi = field_amplitude * np.sin(2*np.pi*x) * polymer_scale
    
    # Compute initial energy density
    energy_density = field.compute_energy_density()
    
    # Find negative energy regions
    negative_indices = np.where(energy_density < 0)[0]
    total_negative_energy = np.sum(energy_density[negative_indices]) if len(negative_indices) > 0 else 0
    
    # Estimate bubble parameters if negative energy exists
    if len(negative_indices) > 0:
        center_idx = negative_indices[np.argmin(energy_density[negative_indices])]
        center_position = x[center_idx]
        bubble_radius = len(negative_indices) * field.dx / 2
        peak_density = np.min(energy_density)
        
        # Create warp bubble object
        bubble = WarpBubble(center_position, bubble_radius, peak_density, polymer_scale)
        stability = bubble.stability_analysis()
    else:
        bubble = None
        stability = None
    
    return {
        "total_negative_energy": total_negative_energy,
        "negative_sites": len(negative_indices),
        "energy_density": energy_density,
        "x_grid": x,
        "bubble": bubble,
        "stability_analysis": stability,
        "polymer_enhancement": polymer_scale > 0
    }


def optimize_warp_bubble_parameters(target_negative_energy: float,
                                  max_polymer_scale: float = 1.0,
                                  lattice_size: int = 64) -> Dict:
    """
    Optimize polymer parameters to achieve target negative energy.
    
    Args:
        target_negative_energy: Desired total negative energy
        max_polymer_scale: Maximum polymer parameter to try
        lattice_size: Lattice resolution
        
    Returns:
        Optimization results
    """
    best_params = None
    best_energy = 0
    best_stability = 0
    
    # Parameter scan
    polymer_scales = np.linspace(0.01, max_polymer_scale, 20)
    field_amplitudes = np.linspace(0.5, 2.0, 10)
    
    results = []
    
    for mu_bar in polymer_scales:
        for amplitude in field_amplitudes:
            result = compute_negative_energy_region(lattice_size, mu_bar, amplitude)
            
            neg_energy = result["total_negative_energy"]
            
            if neg_energy < 0:  # We have negative energy
                stability_factor = 1.0
                if result["stability_analysis"]:
                    stability_factor = result["stability_analysis"]["stabilization_factor"]
                
                # Score based on how close to target and stability
                energy_score = min(1.0, abs(neg_energy) / abs(target_negative_energy))
                total_score = energy_score * stability_factor
                
                results.append({
                    "polymer_scale": mu_bar,
                    "field_amplitude": amplitude,
                    "negative_energy": neg_energy,
                    "stability_factor": stability_factor,
                    "score": total_score
                })
                
                if total_score > best_stability:
                    best_params = (mu_bar, amplitude)
                    best_energy = neg_energy
                    best_stability = total_score
    
    return {
        "best_parameters": best_params,
        "best_negative_energy": best_energy,
        "best_stability_score": best_stability,
        "all_results": results,
        "optimization_success": best_params is not None
    }


def ford_roman_violation_analysis(bubble: WarpBubble, observation_time: float) -> Dict:
    """
    Analyze Ford-Roman quantum inequality violations.
    
    The Ford-Roman bound states that negative energy cannot persist for
    longer than τ ~ (Δx)²/|ρ_neg| in the classical case.
    
    Args:
        bubble: WarpBubble instance
        observation_time: Time period to analyze
        
    Returns:
        Violation analysis results
    """
    # Classical Ford-Roman bound
    classical_bound = (bubble.radius**2) / (abs(bubble.rho_neg) + 1e-10)
    
    # Polymer-modified bound (theoretical prediction)
    if bubble.mu_bar > 0:
        polymer_factor = 1 + bubble.mu_bar**2 / (1 + bubble.mu_bar**4)
        polymer_bound = classical_bound * polymer_factor
    else:
        polymer_bound = classical_bound
    
    # Check violations
    classical_violation = observation_time > classical_bound
    polymer_violation = observation_time > polymer_bound
    
    # Violation strength
    classical_violation_factor = observation_time / classical_bound
    polymer_violation_factor = observation_time / polymer_bound
    
    return {
        "classical_ford_roman_bound": classical_bound,
        "polymer_ford_roman_bound": polymer_bound,
        "observation_time": observation_time,
        "classical_violation": classical_violation,
        "polymer_violation": polymer_violation,
        "classical_violation_factor": classical_violation_factor,
        "polymer_violation_factor": polymer_violation_factor,
        "polymer_enhancement": polymer_bound / classical_bound,
        "violation_possible": classical_violation and not polymer_violation
    }
