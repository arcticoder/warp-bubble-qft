"""
Polymer Field Algebra

Implementation of discrete field commutation relations on a polymer background.
"""

import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class PolymerField:
    """
    A scalar field quantized on a polymer/discrete background.
    
    This implements the polymer representation where:
    - phi_i are field values at lattice sites
    - pi_i are conjugate momenta with polymer-modified commutation relations
    """
    
    def __init__(self, lattice_size: int, polymer_scale: float, mass: float = 1.0):
        """
        Initialize polymer field on a 1D lattice.
        
        Args:
            lattice_size: Number of lattice sites
            polymer_scale: Polymer parameter μ̄ (dimensionless)
            mass: Field mass (default 1.0)
        """
        self.N = lattice_size
        self.mu_bar = polymer_scale
        self.mass = mass
        self.dx = 1.0 / lattice_size  # Lattice spacing
        
        # Initialize field and momentum arrays
        self.phi = np.zeros(lattice_size, dtype=complex)
        self.pi = np.zeros(lattice_size, dtype=complex)
        
        logger.info(f"Initialized PolymerField: N={lattice_size}, μ̄={polymer_scale}")
    
    def set_coherent_state(self, amplitude: float, width: float, center: float = 0.5):
        """Set field to a Gaussian coherent state."""
        x = np.linspace(0, 1, self.N)
        self.phi = amplitude * np.exp(-(x - center)**2 / (2 * width**2))
        self.pi = np.zeros_like(self.phi)  # Start with zero momentum
    
    def polymer_momentum_operator(self, p_classical: np.ndarray) -> np.ndarray:
        """
        Apply polymer modification to momentum operator.
        
        Classical: π → π
        Polymer: π → sin(μ̄π)/μ̄
        """
        if self.mu_bar == 0:
            return p_classical
        return np.sin(self.mu_bar * p_classical) / self.mu_bar
    
    def compute_energy_density(self) -> np.ndarray:
        """
        Compute local energy density T^00.
        
        Returns:
            Array of energy density values at each lattice site
        """
        # Kinetic term (with polymer modification)
        kinetic = 0.5 * self.polymer_momentum_operator(self.pi)**2
        
        # Gradient term (discrete derivative)
        grad_phi = np.gradient(self.phi, self.dx)
        gradient = 0.5 * np.abs(grad_phi)**2
        
        # Mass term
        potential = 0.5 * self.mass**2 * np.abs(self.phi)**2
        
        return kinetic + gradient + potential
    
    def evolve_step(self, dt: float):
        """
        Evolve field by one time step using polymer-modified equations.
        
        Classical evolution:
        ∂φ/∂t = π
        ∂π/∂t = ∇²φ - m²φ
        
        Polymer evolution:
        ∂φ/∂t = sin(μ̄π)/μ̄  
        ∂π/∂t = ∇²φ - m²φ
        """
        # Store current values
        phi_old = self.phi.copy()
        pi_old = self.pi.copy()
        
        # Update phi using polymer-modified momentum
        dphi_dt = self.polymer_momentum_operator(pi_old)
        self.phi = phi_old + dt * dphi_dt
        
        # Update pi using discrete Laplacian
        laplacian = (np.roll(phi_old, 1) - 2*phi_old + np.roll(phi_old, -1)) / self.dx**2
        dpi_dt = laplacian - self.mass**2 * phi_old
        self.pi = pi_old + dt * dpi_dt


def compute_commutator(i: int, j: int, polymer_scale: float = 0.0) -> complex:
    """
    Compute the commutator [φ_i, π_j] in the polymer representation.
    
    Classical: [φ_i, π_j] = iℏδ_ij
    Polymer: Modified by sin functions
    
    Args:
        i, j: Lattice indices
        polymer_scale: Polymer parameter μ̄
        
    Returns:
        Commutator value
    """
    if i != j:
        return 0.0
    
    if polymer_scale == 0:
        return 1j  # Classical result (ℏ=1)
    
    # Polymer modification (simplified)
    # Full treatment requires careful handling of operator ordering
    return 1j * np.sinc(polymer_scale)  # sinc(x) = sin(πx)/(πx)


def analyze_negative_energy_formation(field: PolymerField, time_steps: int = 100) -> dict:
    """
    Analyze conditions for negative energy density formation.
    
    Args:
        field: PolymerField instance
        time_steps: Number of evolution steps
        
    Returns:
        Dictionary with analysis results
    """
    dt = 0.01
    negative_energy_history = []
    total_energy_history = []
    
    for step in range(time_steps):
        energy_density = field.compute_energy_density()
        
        # Check for negative energy regions
        negative_sites = np.where(energy_density < 0)[0]
        negative_energy = np.sum(energy_density[negative_sites]) if len(negative_sites) > 0 else 0
        total_energy = np.sum(energy_density)
        
        negative_energy_history.append(negative_energy)
        total_energy_history.append(total_energy)
        
        # Evolve field
        field.evolve_step(dt)
    
    # Find longest negative energy period
    negative_periods = []
    current_period = 0
    
    for neg_energy in negative_energy_history:
        if neg_energy < 0:
            current_period += 1
        else:
            if current_period > 0:
                negative_periods.append(current_period)
            current_period = 0
    
    if current_period > 0:
        negative_periods.append(current_period)
    
    max_negative_period = max(negative_periods) if negative_periods else 0
    
    return {
        "max_negative_energy": min(negative_energy_history),
        "max_negative_duration": max_negative_period * dt,
        "total_negative_periods": len(negative_periods),
        "energy_conservation": abs(total_energy_history[-1] - total_energy_history[0]),
        "negative_energy_history": negative_energy_history,
        "total_energy_history": total_energy_history
    }
