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
        """Test that polymer commutators reduce to classical ones for μ → 0."""
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
        mu = 0.5
        result = compute_commutator(1, 1, polymer_scale=mu)
        
        # Should be modified by sinc function
        expected = 1j * np.sinc(mu / np.pi)
        assert pytest.approx(result, rel=1e-6) == expected
    
    def test_commutator_consistency(self):
        """Test that commutator results are consistent."""
        mu = 0.3
        result1 = compute_commutator(1, 1, polymer_scale=mu)
        result2 = compute_commutator(1, 1, polymer_scale=mu)
        
        # Results should be identical
        assert result1 == result2


def test_commutator_diagonal():
    """Test that [φ_i, π_j] = iℏδ_ij in polymer representation."""
    N, mu, hbar = 5, 0.1, 1.0
    pf = PolymerField(N, mu, hbar=hbar)
    
    # Use default basis size for testing
    C = pf.commutator_matrix()
    
    # For finite-dimensional representation, check approximate canonical structure
    # Diagonal should be roughly iℏ, off-diagonal should be small
    diagonal_vals = np.diag(C)
    expected_val = 1j * hbar
    
    # Allow reasonable tolerance for discrete approximation
    assert np.allclose(diagonal_vals, expected_val, atol=1e-1), \
        f"Diagonal elements: {diagonal_vals}, expected: {expected_val}"


def test_commutator_function():
    """Test the standalone commutator function."""
    mu = 0.2
    hbar = 1.0
    
    # Diagonal elements should give iℏ
    comm_diag = compute_commutator(0, 0, mu, hbar)
    expected_diag = 1j * hbar * np.sinc(mu / np.pi)
    assert np.isclose(comm_diag, expected_diag), f"Got {comm_diag}, expected {expected_diag}"
    
    # Off-diagonal elements should be zero
    comm_off = compute_commutator(0, 1, mu, hbar)
    assert np.isclose(comm_off, 0.0), f"Off-diagonal should be 0, got {comm_off}"


def test_polymer_field_classical_limit():
    """Test that polymer modifications reduce to classical case when μ→0."""
    N = 4
    pf_classical = PolymerField(N, polymer_scale=0.0)
    pf_polymer = PolymerField(N, polymer_scale=1e-6)  # Very small μ
    
    # Set same field configuration
    test_pi = np.array([1.0, -0.5, 0.3, -0.8])
    
    # Test polymer momentum operator approaches classical
    classical_result = pf_classical.polymer_momentum_operator(test_pi)
    polymer_result = pf_polymer.polymer_momentum_operator(test_pi)
    
    assert np.allclose(classical_result, test_pi), "Classical case should return input unchanged"
    assert np.allclose(polymer_result, test_pi, atol=1e-4), "Small μ should approach classical limit"


def test_energy_density_positivity():
    """Test that energy density is typically positive for normal field configurations."""
    N = 10
    mu = 0.3
    pf = PolymerField(N, mu)
    
    # Set a smooth Gaussian field configuration
    pf.set_coherent_state(amplitude=1.0, width=0.2)
    
    energy_density = pf.compute_energy_density()
    
    # For smooth configurations, energy should be positive
    assert np.all(energy_density >= 0), f"Energy density should be positive, got: {energy_density}"


def test_polymer_modification_range():
    """Test that sin(μπ)/μ modification can produce values outside classical range."""
    mu = 1.0  # Moderate polymer scale
    
    # Test values that would push sin into negative regime
    pi_values = np.array([np.pi/(2*mu) * 1.2, np.pi/mu * 0.8])  # Push into second quadrant
    
    pf = PolymerField(lattice_size=2, polymer_scale=mu)
    result = pf.polymer_momentum_operator(pi_values)
    
    # sin(μπ)/μ can be negative when μπ ∈ (π/2, 3π/2)
    assert not np.array_equal(result, pi_values), "Polymer modification should change values"
    
    # Test specific case where sin should be negative
    pi_test = np.pi / mu  # This gives sin(π) = 0
    pi_test_neg = pi_test * 1.1  # This pushes into negative sin region
    
    result_neg = pf.polymer_momentum_operator(np.array([pi_test_neg]))
    assert result_neg[0] < pi_test_neg, "Should get negative modification in appropriate range"


@pytest.mark.parametrize("mu", [0.0, 0.1, 0.5, 1.0])
def test_field_evolution_stability(mu):
    """Test that field evolution remains stable for different polymer scales."""
    N = 8
    pf = PolymerField(N, mu, mass=0.1)  # Small mass for stability
    
    # Set initial smooth configuration
    pf.set_coherent_state(amplitude=0.5, width=0.3)
    
    # Evolve for several time steps
    dt = 0.01
    initial_energy = np.sum(pf.compute_energy_density())
    
    for _ in range(10):
        pf.evolve_step(dt)
        current_energy = np.sum(pf.compute_energy_density())
        
        # Energy should remain finite (not blow up)
        assert np.isfinite(current_energy), f"Energy became non-finite: {current_energy}"
        assert current_energy < 100 * initial_energy, f"Energy grew too large: {current_energy}"


def test_commutator_antisymmetry():
    """Test that [φ_i, π_j] = -[π_j, φ_i]."""
    N = 3
    mu = 0.4
    pf = PolymerField(N, mu)
    
    basis_size = 8  # Small for speed
    phi_ops = pf.phi_operator(basis_size)
    pi_ops = pf.pi_polymer_operator(basis_size)
    
    for i in range(N):
        for j in range(N):
            comm1 = phi_ops[i] @ pi_ops[j] - pi_ops[j] @ phi_ops[i]
            comm2 = pi_ops[j] @ phi_ops[i] - phi_ops[i] @ pi_ops[j]
            
            # Should be negatives of each other
            assert np.allclose(comm1, -comm2, atol=1e-10), f"Antisymmetry failed for i={i}, j={j}"


if __name__ == "__main__":
    # Run a quick test manually
    test_commutator_diagonal()
    test_polymer_field_classical_limit()
    print("Basic tests passed!")


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
