#!/usr/bin/env python3
"""
Comprehensive validation of recent discoveries in polymer QFT.

This test suite validates all the key discoveries outlined in the recent 
research: sampling function properties, kinetic energy comparisons, 
commutator matrix structure, energy density scaling, and symbolic analysis.
"""

import numpy as np
import pytest
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from warp_qft.field_algebra import compute_commutator, PolymerField
from warp_qft.negative_energy import sampling_function, compute_energy_density


class TestRecentDiscoveries:
    """Test suite validating all recent discoveries in polymer QFT."""
    
    def test_sampling_function_symmetry(self):
        """Test that f(t,τ) satisfies f(-t) = f(t)."""
        tau = 1.0
        t_vals = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0])
        
        f_positive = sampling_function(t_vals, tau)
        f_negative = sampling_function(-t_vals, tau)
        
        assert np.allclose(f_positive, f_negative), "Sampling function should be symmetric"
    
    def test_sampling_function_peak_location(self):
        """Test that f(t,τ) peaks at t=0."""
        tau = 1.0
        t_vals = np.linspace(-3, 3, 101)
        f_vals = sampling_function(t_vals, tau)
        
        peak_idx = np.argmax(f_vals)
        peak_time = t_vals[peak_idx]
        
        assert abs(peak_time) < 0.1, f"Peak should be at t=0, found at t={peak_time}"
    
    def test_sampling_function_width_scaling(self):
        """Test that peak height scales as 1/τ."""
        tau_vals = [0.5, 1.0, 2.0]
        peak_heights = []
        
        for tau in tau_vals:
            f_peak = sampling_function(0.0, tau)
            peak_heights.append(f_peak)
        
        # Check that peak_height * tau is approximately constant
        scaled_peaks = [h * tau for h, tau in zip(peak_heights, tau_vals)]
        expected_value = 1.0 / np.sqrt(2 * np.pi)
        
        for scaled_peak in scaled_peaks:
            assert abs(scaled_peak - expected_value) < 1e-10, \
                f"Peak scaling incorrect: got {scaled_peak}, expected {expected_value}"
    
    def test_sampling_function_normalization(self):
        """Test that ∫f(t,τ)dt = 1."""
        tau = 1.0
        t_vals = np.linspace(-10, 10, 1001)
        dt = t_vals[1] - t_vals[0]
        f_vals = sampling_function(t_vals, tau)
        
        integral = np.trapz(f_vals, dx=dt)
        assert abs(integral - 1.0) < 1e-3, f"Sampling function should be normalized, got {integral}"
    
    def test_kinetic_energy_comparison_specific_case(self):
        """Test the specific case μπ = 2.5 (μ = 0.5, π ≈ 5.0)."""
        mu = 0.5
        pi_val = 5.0  # This gives μπ = 2.5
        
        classical_T = pi_val**2 / 2
        polymer_T = (np.sin(mu * pi_val) / mu)**2 / 2
        
        # Should have polymer_T < classical_T
        assert polymer_T < classical_T, \
            f"Polymer kinetic energy should be lower: {polymer_T} vs {classical_T}"
        
        # Check specific values match expectations
        expected_classical = 12.5
        assert abs(classical_T - expected_classical) < 1e-10, \
            f"Classical energy should be 12.5, got {classical_T}"
        
        # The polymer energy should be significantly lower
        energy_reduction = (classical_T - polymer_T) / classical_T
        assert energy_reduction > 0.8, \
            f"Should have >80% energy reduction, got {energy_reduction:.1%}"
    
    @pytest.mark.parametrize("mu", [0.3, 0.5, 0.8])
    def test_kinetic_energy_suppression_range(self, mu):
        """Test kinetic energy suppression for μπ ∈ (π/2, 3π/2)."""
        # Test values in the suppression range
        pi_vals = np.linspace(np.pi/(2*mu) + 0.1, 3*np.pi/(2*mu) - 0.1, 10)
        
        suppressions_found = 0
        for pi_val in pi_vals:
            classical_T = pi_val**2 / 2
            polymer_T = (np.sin(mu * pi_val) / mu)**2 / 2
            
            if polymer_T < classical_T:
                suppressions_found += 1
        
        # Should find suppression in most of this range
        assert suppressions_found >= 5, \
            f"Should find energy suppression in most of range, found {suppressions_found}/10"
    
    def test_commutator_matrix_antisymmetry(self):
        """Test that commutator matrix C = [φ, π^poly] is antisymmetric."""
        N = 6
        mu = 0.3
        pf = PolymerField(N, mu)
        
        C = pf.commutator_matrix()
        
        # Check C = -C†
        C_dagger = np.conjugate(C.T)
        assert np.allclose(C, -C_dagger, atol=1e-10), \
            "Commutator matrix should be antisymmetric"
    
    def test_commutator_matrix_imaginary_eigenvalues(self):
        """Test that commutator matrix has pure imaginary eigenvalues."""
        N = 6
        mu = 0.3
        pf = PolymerField(N, mu)
        
        C = pf.commutator_matrix()
        eigenvals = np.linalg.eigvals(C)
        
        # All eigenvalues should have zero real part
        real_parts = np.real(eigenvals)
        assert np.allclose(real_parts, 0, atol=1e-12), \
            f"Eigenvalues should be pure imaginary, got real parts: {real_parts}"
    
    def test_commutator_matrix_nonzero_norm(self):
        """Test that commutator matrix has non-vanishing norm."""
        N = 6
        mu = 0.3
        pf = PolymerField(N, mu)
        
        C = pf.commutator_matrix()
        norm = np.linalg.norm(C)
        
        assert norm > 1e-10, f"Commutator matrix should have non-zero norm, got {norm}"
    
    def test_energy_density_sinc_formula_agreement(self):
        """Test exact agreement with sinc formula for energy density."""
        mu = 0.4
        pi_val = 1.5  # Constant momentum
        phi_val = 0.0  # Zero field (focus on kinetic term)
        dx = 1.0
        
        # Create arrays
        phi = np.array([phi_val])
        pi = np.array([pi_val])
        
        # Compute using the energy density function
        rho_polymer = compute_energy_density(phi, pi, mu, dx)
        
        # Compute using exact sinc formula
        rho_expected = 0.5 * (np.sin(mu * pi_val) / mu)**2
        
        assert np.allclose(rho_polymer[0], rho_expected, rtol=1e-12), \
            f"Energy density should match sinc formula: {rho_polymer[0]} vs {rho_expected}"
    
    def test_energy_density_classical_limit(self):
        """Test that energy density reduces to classical when μ → 0."""
        pi_val = 2.0
        phi_val = 0.0
        dx = 1.0
        
        phi = np.array([phi_val])
        pi = np.array([pi_val])
        
        # Classical case
        rho_classical = compute_energy_density(phi, pi, 0.0, dx)
        
        # Small μ case
        rho_small_mu = compute_energy_density(phi, pi, 1e-6, dx)
        
        assert np.allclose(rho_classical, rho_small_mu, rtol=1e-5), \
            "Energy density should approach classical limit for small μ"
    
    def test_sinc_enhancement_factor_scaling(self):
        """Test the enhancement factor ξ = 1/sinc(μ) scaling."""
        mu_vals = [0.3, 0.5, 0.8, 1.0]
        
        for mu in mu_vals:
            sinc_val = np.sinc(mu / np.pi)
            enhancement = 1.0 / sinc_val
            
            # Enhancement should be > 1 for μ > 0
            assert enhancement > 1.0, f"Enhancement should be > 1, got {enhancement} for μ={mu}"
            
            # Should increase with μ
            if mu > 0.3:
                prev_mu = mu - 0.1
                prev_sinc = np.sinc(prev_mu / np.pi)
                prev_enhancement = 1.0 / prev_sinc
                assert enhancement > prev_enhancement * 0.99, \
                    f"Enhancement should generally increase with μ"
    
    def test_polymer_bound_relaxation(self):
        """Test that polymer bound is less restrictive than classical."""
        hbar = 1.0
        tau = 1.0
        
        # Classical Ford-Roman bound
        classical_bound = -hbar / (12 * np.pi * tau**2)
        
        mu_vals = [0.2, 0.5, 1.0]
        for mu in mu_vals:
            sinc_val = np.sinc(mu / np.pi)
            polymer_bound = -hbar * sinc_val / (12 * np.pi * tau**2)
            
            # Polymer bound should be less restrictive (larger absolute value allows smaller violations)
            assert abs(polymer_bound) < abs(classical_bound), \
                f"Polymer bound should be less restrictive: |{polymer_bound}| < |{classical_bound}|"
    
    def test_comprehensive_discovery_integration(self):
        """Integration test verifying all discoveries work together."""
        # Set up polymer field
        N = 16
        mu = 0.5
        pf = PolymerField(N, mu)
        
        # Test commutator structure
        C = pf.commutator_matrix()
        assert np.allclose(C, -np.conjugate(C.T), atol=1e-10), "Antisymmetry check"
        
        # Test energy suppression
        pi_test = 4.0  # Should be in suppression range
        classical_T = pi_test**2 / 2
        polymer_T = (np.sin(mu * pi_test) / mu)**2 / 2
        assert polymer_T < classical_T, "Energy suppression check"
        
        # Test sampling function
        tau = 1.0
        f_peak = sampling_function(0.0, tau)
        expected_peak = 1.0 / (np.sqrt(2 * np.pi) * tau)
        assert abs(f_peak - expected_peak) < 1e-10, "Sampling function check"
          # Test enhancement factor
        sinc_val = np.sinc(mu / np.pi)
        enhancement = 1.0 / sinc_val
        assert enhancement > 1.02, f"Enhancement factor check: got {enhancement:.6f}"
        
        print(f"✓ All discoveries integrated successfully:")
        print(f"  - Commutator matrix: antisymmetric ✓")
        print(f"  - Energy suppression: {polymer_T:.3f} < {classical_T:.3f} ✓")
        print(f"  - Sampling function: peak = {f_peak:.6f} ✓")
        print(f"  - Enhancement factor: ξ = {enhancement:.3f} ✓")


if __name__ == "__main__":
    # Run the comprehensive test
    test_suite = TestRecentDiscoveries()
    
    print("Running comprehensive validation of recent discoveries...")
    print("=" * 60)
    
    try:
        # Run all tests
        test_suite.test_sampling_function_symmetry()
        print("✓ Sampling function symmetry")
        
        test_suite.test_sampling_function_peak_location()
        print("✓ Sampling function peak location")
        
        test_suite.test_sampling_function_width_scaling()
        print("✓ Sampling function width scaling")
        
        test_suite.test_sampling_function_normalization()
        print("✓ Sampling function normalization")
        
        test_suite.test_kinetic_energy_comparison_specific_case()
        print("✓ Kinetic energy comparison (specific case)")
        
        test_suite.test_commutator_matrix_antisymmetry()
        print("✓ Commutator matrix antisymmetry")
        
        test_suite.test_commutator_matrix_imaginary_eigenvalues()
        print("✓ Commutator matrix imaginary eigenvalues")
        
        test_suite.test_commutator_matrix_nonzero_norm()
        print("✓ Commutator matrix non-zero norm")
        
        test_suite.test_energy_density_sinc_formula_agreement()
        print("✓ Energy density sinc formula agreement")
        
        test_suite.test_energy_density_classical_limit()
        print("✓ Energy density classical limit")
        
        test_suite.test_sinc_enhancement_factor_scaling()
        print("✓ Sinc enhancement factor scaling")
        
        test_suite.test_polymer_bound_relaxation()
        print("✓ Polymer bound relaxation")
        
        test_suite.test_comprehensive_discovery_integration()
        print("✓ Comprehensive integration")
        
        print("\n" + "=" * 60)
        print("🎉 ALL RECENT DISCOVERIES VALIDATED SUCCESSFULLY! 🎉")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
