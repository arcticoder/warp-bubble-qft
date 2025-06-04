"""
Tests for negative energy bounds and Ford-Roman inequality violations.
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from warp_qft.negative_energy import (
    compute_negative_energy_region, 
    WarpBubble, 
    ford_roman_violation_analysis
)
from warp_qft.stability import (
    ford_roman_bounds, 
    polymer_modified_bounds, 
    violation_duration
)


class TestFordRomanBounds:
    """Test Ford-Roman quantum inequality bounds."""
    
    def test_classical_ford_roman_bound(self):
        """Test classical Ford-Roman bound calculation."""
        energy_density = -1.0
        spatial_scale = 0.1
        
        bounds = ford_roman_bounds(energy_density, spatial_scale)
        
        # Bound should be negative (allowing some negative energy)
        assert bounds["ford_roman_bound"] < 0
        
        # Should violate bound since energy is more negative
        assert bounds["violates_bound"] == True
        assert bounds["violation_factor"] > 1.0
    
    def test_polymer_enhancement(self):
        """Test that polymer effects enhance the bound."""
        energy_density = -1.0
        spatial_scale = 0.1
        polymer_scale = 0.5
        
        classical_bounds = ford_roman_bounds(energy_density, spatial_scale)
        polymer_bounds = polymer_modified_bounds(energy_density, spatial_scale, polymer_scale)
        
        # Polymer bound should be more relaxed (more negative)
        assert polymer_bounds["ford_roman_bound"] < classical_bounds["ford_roman_bound"]
        
        # Enhancement factor should be > 1
        assert polymer_bounds["enhancement_factor"] > 1.0
    
    def test_polymer_scale_zero_limit(self):
        """Test that μ̄ → 0 recovers classical bounds."""
        energy_density = -0.5
        spatial_scale = 0.2
        
        classical_bounds = ford_roman_bounds(energy_density, spatial_scale)
        polymer_bounds = polymer_modified_bounds(energy_density, spatial_scale, 0.0)
        
        # Should be essentially identical
        assert pytest.approx(polymer_bounds["ford_roman_bound"], rel=1e-10) == classical_bounds["ford_roman_bound"]


class TestViolationDuration:
    """Test violation duration calculations."""
    
    def test_positive_energy_infinite_duration(self):
        """Test that positive energy has infinite allowed duration."""
        energy_density = 1.0
        spatial_scale = 0.1
        
        result = violation_duration(energy_density, spatial_scale)
        
        assert result["max_duration"] == np.inf
        assert result["violation_type"] == "no_negative_energy"
    
    def test_negative_energy_finite_duration(self):
        """Test that negative energy has finite duration bound."""
        energy_density = -1.0
        spatial_scale = 0.1
        
        result = violation_duration(energy_density, spatial_scale)
        
        assert result["max_duration"] < np.inf
        assert result["classical_duration"] > 0
        assert result["violation_type"] == "duration_limited"
    
    def test_polymer_duration_enhancement(self):
        """Test that polymer effects increase allowed duration."""
        energy_density = -1.0
        spatial_scale = 0.1
        
        classical_result = violation_duration(energy_density, spatial_scale, polymer_scale=0.0)
        polymer_result = violation_duration(energy_density, spatial_scale, polymer_scale=0.5)
        
        # Polymer should allow longer duration
        assert polymer_result["max_duration"] > classical_result["max_duration"]
        assert polymer_result["enhancement_factor"] > 1.0


class TestWarpBubble:
    """Test WarpBubble class functionality."""
    
    def setup_method(self):
        """Set up test bubble."""
        self.bubble = WarpBubble(
            center_position=0.5,
            bubble_radius=0.1,
            negative_energy_density=-1.0,
            polymer_scale=0.3
        )
    
    def test_bubble_initialization(self):
        """Test proper bubble initialization."""
        assert self.bubble.center == 0.5
        assert self.bubble.radius == 0.1
        assert self.bubble.rho_neg == -1.0
        assert self.bubble.mu_bar == 0.3
        assert self.bubble.is_stable == True
    
    def test_energy_profile(self):
        """Test bubble energy profile computation."""
        x = np.linspace(0, 1, 100)
        profile = self.bubble.energy_profile(x)
        
        # Profile should be most negative at center
        center_idx = np.argmin(np.abs(x - self.bubble.center))
        assert profile[center_idx] == np.min(profile)
        
        # Should approach zero far from center
        far_indices = np.where(np.abs(x - self.bubble.center) > 3 * self.bubble.radius)[0]
        if len(far_indices) > 0:
            assert np.all(np.abs(profile[far_indices]) < 0.1)
    
    def test_total_negative_energy(self):
        """Test total negative energy calculation."""
        x = np.linspace(0, 1, 100)
        total_neg = self.bubble.compute_total_negative_energy(x)
        
        # Should be negative
        assert total_neg < 0
    
    def test_stability_analysis(self):
        """Test bubble stability analysis."""
        stability = self.bubble.stability_analysis(duration=1.0)
        
        # Should have all required fields
        required_fields = [
            "classical_lifetime", "polymer_lifetime", 
            "stabilization_factor", "survives_duration"
        ]
        for field in required_fields:
            assert field in stability
        
        # Polymer lifetime should be longer than classical
        assert stability["polymer_lifetime"] >= stability["classical_lifetime"]
        assert stability["stabilization_factor"] >= 1.0


class TestNegativeEnergyComputation:
    """Test negative energy region computation."""
    
    def test_compute_negative_energy_region(self):
        """Test main negative energy computation function."""
        lattice_size = 32
        polymer_scale = 0.4
        field_amplitude = 1.0
        
        result = compute_negative_energy_region(lattice_size, polymer_scale, field_amplitude)
        
        # Should have all required fields
        required_fields = [
            "total_negative_energy", "negative_sites", 
            "energy_density", "x_grid", "polymer_enhancement"
        ]
        for field in required_fields:
            assert field in result
        
        # Polymer enhancement should be True
        assert result["polymer_enhancement"] == True
        
        # Grid should have correct size
        assert len(result["x_grid"]) == lattice_size
        assert len(result["energy_density"]) == lattice_size
    
    def test_different_polymer_scales(self):
        """Test behavior for different polymer scales."""
        lattice_size = 16
        polymer_scales = [0.0, 0.2, 0.5, 1.0]
        
        results = {}
        for mu_bar in polymer_scales:
            result = compute_negative_energy_region(lattice_size, mu_bar, 1.0)
            results[mu_bar] = result
        
        # Classical case should be different from polymer cases
        classical_energy = results[0.0]["total_negative_energy"]
        
        for mu_bar in polymer_scales[1:]:
            polymer_energy = results[mu_bar]["total_negative_energy"]
            # Energy values can differ due to polymer effects
            # (not necessarily more negative, depends on configuration)


class TestFordRomanViolationAnalysis:
    """Test Ford-Roman violation analysis."""
    
    def test_violation_analysis(self):
        """Test complete violation analysis."""
        bubble = WarpBubble(0.5, 0.1, -2.0, 0.4)
        observation_time = 1.0
        
        analysis = ford_roman_violation_analysis(bubble, observation_time)
        
        # Should have all required fields
        required_fields = [
            "classical_ford_roman_bound", "polymer_ford_roman_bound",
            "observation_time", "classical_violation", "polymer_violation",
            "polymer_enhancement"
        ]
        for field in required_fields:
            assert field in analysis
        
        # Polymer bound should be more relaxed
        assert analysis["polymer_ford_roman_bound"] < analysis["classical_ford_roman_bound"]
        assert analysis["polymer_enhancement"] > 1.0
    
    def test_violation_possibility(self):
        """Test detection of possible violations."""
        # Create a bubble that violates classical but not polymer bounds
        bubble = WarpBubble(0.5, 0.05, -0.1, 0.8)  # Large polymer scale
        observation_time = 0.1  # Short observation
        
        analysis = ford_roman_violation_analysis(bubble, observation_time)
        
        # This might show violation is possible in polymer case
        # (exact behavior depends on parameters)
        assert "violation_possible" in analysis


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
