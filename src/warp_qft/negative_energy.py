"""
Negative Energy and Warp Bubble Formation

Analysis of stable negative energy densities and warp bubble configurations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from .field_algebra import PolymerField
import logging

logger = logging.getLogger(__name__)


def sampling_function(t, tau):
    """Gaussian sampling function of width τ centered at t=0."""
    return np.exp(-t**2/(2*tau**2)) / (np.sqrt(2*np.pi)*tau)


def compute_energy_density(phi, pi, mu, dx):
    """
    phi, pi: arrays of shape (N,) at a single time slice
    mu: polymer scale
    dx: lattice spacing    Returns array ρ_i for i=0…N−1.
    """
    # Kinetic term: [sin(π μ π_i)/(π μ)]^2
    if mu == 0.0:
        # Classical limit: kinetic = π²/2  
        kinetic = pi**2
    else:
        # Polymer-modified kinetic term with corrected sinc
        kinetic = (np.sin(np.pi * mu * pi) / (np.pi * mu))**2
    
    # Gradient term: use periodic boundary for simplicity
    grad = np.roll(phi, -1) - np.roll(phi, 1)
    grad = (grad / (2 * dx))**2
    # Mass term (set m=0 for simplicity or make m a parameter)
    mass = 0.0 * phi**2
    return 0.5 * (kinetic + grad + mass)


def integrate_negative_energy_over_time(N, mu, total_time, dt, dx, tau):
    """
    Create π_i(t) = A exp[-((x_i - x0)^2)/(2 σ^2)] sin(ω t), 
    choose A so that μ π_i(t) enters the regime where sin(μ π) gives lower energy.
    Integrate I = sum_i ∫ ρ_i(t) f(t) dt dx.
    Return I_polymer - I_classical (negative indicates QI violation).
    """
    times = np.arange(-total_time/2, total_time/2, dt)
    x = np.arange(N) * dx
    x0 = N*dx/2
    sigma = N*dx/8
    
    # Choose amplitude to target the regime where polymer energy is lower
    # From analysis: need μπ ≈ 1.5-1.8 for maximum energy reduction
    # Use a consistent amplitude that scales differently with μ
    # to ensure amplitude doesn't overwhelm the effect of μ
    if mu > 0:
        # Use a constant amplitude for fair comparison across different mu values
        A = 2.0  # Fixed amplitude for more consistent behavior
    else:
        A = 1.0  # Classical case
    
    omega = 2*np.pi/total_time

    I_polymer = 0.0
    I_classical = 0.0
    
    for t in times:
        # Build π_i(t): a localized sine‐burst
        pi_t = A * np.exp(-((x-x0)**2)/(2*sigma**2)) * np.sin(omega * t)
        # φ_i(t) remains ~0 for focused kinetic energy test
        phi_t = np.zeros_like(pi_t)

        # Compute polymer energy density
        rho_polymer = compute_energy_density(phi_t, pi_t, mu, dx)
        
        # Compute classical energy density (mu=0)
        rho_classical = compute_energy_density(phi_t, pi_t, 0.0, dx)
        
        f_t = sampling_function(t, tau)
        I_polymer += np.sum(rho_polymer) * f_t * dt * dx
        I_classical += np.sum(rho_classical) * f_t * dt * dx

    # Return the difference: negative means QI violation
    return I_polymer - I_classical


# Example usage (not in library, but in a demo script or test):
if __name__ == "__main__":
    N = 64
    dx = 1.0
    dt = 0.01
    total_time = 10.0
    tau = 1.0
    for mu in [0.0, 0.3, 0.6]:
        I = integrate_negative_energy_over_time(N, mu, total_time, dt, dx, tau)
        print(f"μ={mu:.2f}: ∫ρ f dt dx = {I:.6f}")


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
    dx = 1.0 / lattice_size  # Default spacing
    field = PolymerField(lattice_size, polymer_scale, dx)
    
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
    # Import stability functions
    from warp_qft.bubble_stability import ford_roman_violation_analysis as analyze_violation
    
    # Use the implementation from bubble_stability.py
    return analyze_violation(bubble, observation_time)


def compute_negative_energy_region(bubble: WarpBubble) -> Dict:
    """
    Compute the spatial region where negative energy density exists.
    
    Args:
        bubble: WarpBubble instance
        
    Returns:
        Dictionary with negative energy region analysis
    """
    # Get energy density profile
    rho = bubble.energy_density()
    x = np.linspace(-bubble.radius, bubble.radius, len(rho))
    
    # Find negative energy regions
    negative_mask = rho < 0
    if not np.any(negative_mask):
        return {
            "has_negative_region": False,
            "negative_volume": 0.0,
            "peak_negative_density": 0.0
        }
    
    # Compute negative energy volume
    negative_volume = np.sum(negative_mask) * (x[1] - x[0])
    peak_negative_density = np.min(rho[negative_mask])
    
    return {
        "has_negative_region": True,
        "negative_volume": negative_volume,
        "peak_negative_density": peak_negative_density,
        "negative_indices": np.where(negative_mask)[0]
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
    dx = 1.0 / lattice_size  # Default spacing
    field = PolymerField(lattice_size, polymer_scale, dx)
    
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
