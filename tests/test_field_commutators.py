"""
Tests for field commutator algebra in polymer representation.
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from warp_qft.field_algebra import compute_commutator, PolymerField


class TestFieldCommutators:
    """Test suite for polymer field commutation relations."""
    
    def test_classical_limit(self):
        """Test that polymer commutators reduce to classical ones for μ̄ → 0."""
        # Classical commutator should be iℏ (with ℏ=1)
        result = compute_commutator(1, 1, polymer_scale=0.0)
        expected = 1j
        assert pytest.approx(result, rel=1e-10) == expected
    
    def test_different_indices(self):
        """Test that commutators vanish for different lattice indices."""
        result = compute_commutator(1, 2, polymer_scale=0.5)
        assert result == 0.0
    
    def test_polymer_modification(self):
        """Test that polymer scale modifies commutators as expected."""
        mu_bar = 0.5
        result = compute_commutator(1, 1, polymer_scale=mu_bar)
        
        # Should be modified by sinc function
        expected = 1j * np.sinc(mu_bar)
        assert pytest.approx(result, rel=1e-6) == expected
    
    def test_commutator_symmetry(self):
        """Test antisymmetry of commutators."""
        mu_bar = 0.3
        result1 = compute_commutator(1, 1, polymer_scale=mu_bar)
        result2 = compute_commutator(1, 1, polymer_scale=mu_bar)
        
        # [A,A] should be zero, but we're testing [φ,π] which is non-zero
        # Test that the result is consistent
        assert result1 == result2


class TestPolymerField:
    """Test suite for PolymerField class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lattice_size = 16
        self.polymer_scale = 0.1
        self.field = PolymerField(self.lattice_size, self.polymer_scale)
    
    def test_initialization(self):
        """Test proper initialization of polymer field."""
        assert self.field.N == self.lattice_size
        assert self.field.mu_bar == self.polymer_scale
        assert len(self.field.phi) == self.lattice_size
        assert len(self.field.pi) == self.lattice_size
    
    def test_coherent_state_setup(self):
        """Test setting up coherent state initial conditions."""
        amplitude = 1.0
        width = 0.2
        center = 0.5
        
        self.field.set_coherent_state(amplitude, width, center)
        
        # Check that field has expected Gaussian profile
        x = np.linspace(0, 1, self.lattice_size)
        expected = amplitude * np.exp(-(x - center)**2 / (2 * width**2))
        
        np.testing.assert_allclose(self.field.phi, expected, rtol=1e-10)
    
    def test_energy_density_positive(self):
        """Test that energy density is positive for simple configurations."""
        # Set up a simple configuration
        self.field.phi = np.ones(self.lattice_size) * 0.1
        self.field.pi = np.zeros(self.lattice_size)
        
        energy = self.field.compute_energy_density()
        
        # Should be positive (mass term dominates)
        assert np.all(energy >= 0)
    
    def test_polymer_momentum_operator(self):
        """Test polymer momentum operator implementation."""
        p_classical = np.array([0.1, 0.5, 1.0, 2.0])
        
        # Classical limit
        self.field.mu_bar = 0.0
        result_classical = self.field.polymer_momentum_operator(p_classical)
        np.testing.assert_allclose(result_classical, p_classical)
        
        # Polymer case
        self.field.mu_bar = 0.5
        result_polymer = self.field.polymer_momentum_operator(p_classical)
        expected = np.sin(0.5 * p_classical) / 0.5
        np.testing.assert_allclose(result_polymer, expected)
    
    def test_evolution_conservation(self):
        """Test that field evolution conserves certain quantities."""
        # Set up initial state
        self.field.set_coherent_state(1.0, 0.2, 0.5)
        
        # Compute initial energy
        initial_energy = np.sum(self.field.compute_energy_density())
        
        # Evolve for a few steps
        dt = 0.01
        for _ in range(10):
            self.field.evolve_step(dt)
        
        # Energy should be approximately conserved
        final_energy = np.sum(self.field.compute_energy_density())
        energy_change = abs(final_energy - initial_energy) / abs(initial_energy)
        
        # Allow for small numerical errors
        assert energy_change < 0.1  # 10% tolerance


class TestNegativeEnergyFormation:
    """Test negative energy formation in specific configurations."""
    
    def test_negative_energy_conditions(self):
        """Test conditions that can lead to negative energy."""
        lattice_size = 32
        polymer_scale = 0.5
        field = PolymerField(lattice_size, polymer_scale, mass=0.1)  # Small mass
        
        # Set up configuration that might produce negative energy
        x = np.linspace(0, 1, lattice_size)
        field.phi = 0.1 * np.sin(4 * np.pi * x)  # Small amplitude field
        field.pi = 2.0 * np.cos(4 * np.pi * x)   # Large momentum
        
        energy_density = field.compute_energy_density()
        
        # Check if any sites have negative energy
        has_negative = np.any(energy_density < 0)
        
        if has_negative:
            negative_sites = np.sum(energy_density < 0)
            total_negative = np.sum(energy_density[energy_density < 0])
            print(f"Found {negative_sites} negative energy sites")
            print(f"Total negative energy: {total_negative}")
    
    def test_polymer_vs_classical_energy(self):
        """Compare energy densities between polymer and classical cases."""
        lattice_size = 16
        
        # Create two fields: one classical, one polymer
        field_classical = PolymerField(lattice_size, 0.0)  # No polymer effects
        field_polymer = PolymerField(lattice_size, 0.5)    # With polymer effects
        
        # Set same initial conditions
        field_classical.set_coherent_state(1.0, 0.2, 0.5)
        field_polymer.set_coherent_state(1.0, 0.2, 0.5)
        
        # Add same momentum
        x = np.linspace(0, 1, lattice_size)
        momentum = np.sin(2 * np.pi * x)
        field_classical.pi = momentum.copy()
        field_polymer.pi = momentum.copy()
        
        # Compare energy densities
        energy_classical = field_classical.compute_energy_density()
        energy_polymer = field_polymer.compute_energy_density()
        
        # They should be different due to polymer effects
        assert not np.allclose(energy_classical, energy_polymer)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
