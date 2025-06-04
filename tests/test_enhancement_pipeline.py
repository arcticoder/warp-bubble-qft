"""
Unit tests for the enhancement pipeline modules.

Tests all major components:
- LQG profiles and optimal parameters
- Metric backreaction calculations  
- Enhancement pathways (cavity, squeezing, multi-bubble)
- Complete pipeline orchestration
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.warp_qft.lqg_profiles import (
    toy_negative_energy, lqg_negative_energy, optimal_lqg_parameters,
    compare_profile_types, scan_lqg_parameter_space
)
from src.warp_qft.backreaction_solver import (
    BackreactionSolver, apply_backreaction_correction
)
from src.warp_qft.enhancement_pathway import (
    EnhancementConfig, CavityBoostCalculator, QuantumSqueezingEnhancer,
    MultiBubbleSuperposition, EnhancementPathwayOrchestrator
)
from src.warp_qft.enhancement_pipeline import (
    PipelineConfig, WarpBubbleEnhancementPipeline,
    run_quick_feasibility_check, find_first_unity_configuration
)


class TestLQGProfiles(unittest.TestCase):
    """Test LQG-corrected negative energy profiles."""
    
    def setUp(self):
        self.mu_test = 0.10
        self.R_test = 2.3
        self.tolerance = 1e-10
    
    def test_toy_negative_energy(self):
        """Test toy model negative energy calculation."""
        energy = toy_negative_energy(self.mu_test, self.R_test)
        
        self.assertGreater(energy, 0)  # Should return positive magnitude
        self.assertIsInstance(energy, float)
        
        # Test with different parameters
        energy_small = toy_negative_energy(0.05, 1.0)
        energy_large = toy_negative_energy(0.20, 4.0)
        
        self.assertGreater(energy_small, 0)
        self.assertGreater(energy_large, 0)
    
    def test_lqg_enhancement_factors(self):
        """Test LQG enhancement factors are correct."""
        toy_energy = toy_negative_energy(self.mu_test, self.R_test)
        
        # Test different LQG prescriptions
        bojo_energy = lqg_negative_energy(self.mu_test, self.R_test, "bojo")
        ashtekar_energy = lqg_negative_energy(self.mu_test, self.R_test, "ashtekar")
        polymer_energy = lqg_negative_energy(self.mu_test, self.R_test, "polymer_field")
        
        # Check enhancement factors
        self.assertAlmostEqual(bojo_energy / toy_energy, 2.1, places=1)
        self.assertAlmostEqual(ashtekar_energy / toy_energy, 1.8, places=1)
        self.assertAlmostEqual(polymer_energy / toy_energy, 2.3, places=1)
        
        # Polymer field should be the best
        self.assertGreater(polymer_energy, bojo_energy)
        self.assertGreater(polymer_energy, ashtekar_energy)
    
    def test_optimal_lqg_parameters(self):
        """Test optimal LQG parameters are reasonable."""
        optimal = optimal_lqg_parameters()
        
        self.assertIn("mu_optimal", optimal)
        self.assertIn("R_optimal", optimal) 
        self.assertIn("enhancement_factor", optimal)
        
        # Check values are in expected ranges
        self.assertGreater(optimal["mu_optimal"], 0.05)
        self.assertLess(optimal["mu_optimal"], 0.25)
        self.assertGreater(optimal["R_optimal"], 1.0)
        self.assertLess(optimal["R_optimal"], 5.0)
        self.assertGreater(optimal["enhancement_factor"], 1.5)
    
    def test_profile_comparison(self):
        """Test profile comparison functionality."""
        comparison = compare_profile_types(self.mu_test, self.R_test)
        
        self.assertIn("toy_model", comparison)
        self.assertIn("polymer_field", comparison)
        
        # LQG profiles should enhance over toy model
        toy_energy = comparison["toy_model"]
        for profile, energy in comparison.items():
            if profile != "toy_model":
                self.assertGreater(energy, toy_energy)
    
    def test_parameter_space_scan(self):
        """Test parameter space scanning."""
        mu_range = np.linspace(0.08, 0.12, 5)
        R_range = np.linspace(2.0, 2.6, 5)
        
        scan_results = scan_lqg_parameter_space(mu_range, R_range)
        
        self.assertIn("mu_optimal", scan_results)
        self.assertIn("R_optimal", scan_results)
        self.assertIn("max_enhancement", scan_results)
        self.assertIn("enhancement_grid", scan_results)
        
        # Check grid dimensions
        grid = scan_results["enhancement_grid"]
        self.assertEqual(grid.shape, (len(mu_range), len(R_range)))
        
        # Check optimal values are in range
        self.assertIn(scan_results["mu_optimal"], mu_range)
        self.assertIn(scan_results["R_optimal"], R_range)


class TestBackreactionSolver(unittest.TestCase):
    """Test metric backreaction calculations."""
    
    def setUp(self):
        self.solver = BackreactionSolver(grid_size=100)
        self.R_bubble = 2.0
        
    def test_solver_initialization(self):
        """Test backreaction solver initialization."""
        self.assertEqual(self.solver.grid_size, 100)
        self.assertEqual(self.solver.tolerance, 1e-6)
    
    def test_spatial_grid_setup(self):
        """Test spatial grid generation."""
        grid = self.solver.setup_spatial_grid(5.0)
        
        self.assertEqual(len(grid), self.solver.grid_size)
        self.assertAlmostEqual(grid[0], -5.0, places=5)
        self.assertAlmostEqual(grid[-1], 5.0, places=5)
    
    def test_initial_metric_guess(self):
        """Test initial metric guess generation."""
        r = self.solver.setup_spatial_grid(5.0)
        g_tt, g_rr = self.solver.initial_metric_guess(r, self.R_bubble)
        
        self.assertEqual(len(g_tt), len(r))
        self.assertEqual(len(g_rr), len(r))
        
        # Time component should be approximately -1 (Minkowski)
        self.assertTrue(np.allclose(g_tt, -1.0, atol=0.1))
        
        # Spatial component should be approximately 1 (with perturbations)
        self.assertTrue(np.allclose(g_rr, 1.0, atol=0.2))
    
    def test_stress_energy_tensor(self):
        """Test stress-energy tensor calculation."""
        r = np.linspace(-3, 3, 50)
        rho_neg = -np.exp(-(r**2) / 2)  # Gaussian negative energy
        g_tt = -np.ones_like(r)
        g_rr = np.ones_like(r)
        
        stress_energy = self.solver.stress_energy_tensor(r, rho_neg, g_tt, g_rr)
        
        self.assertIn("T_00", stress_energy)
        self.assertIn("T_rr", stress_energy)
        
        # Energy density should be negative
        self.assertTrue(np.all(stress_energy["T_00"] <= 0))
    
    def test_apply_backreaction_correction_quick(self):
        """Test quick backreaction correction."""
        original_energy = 1.0
        
        def test_profile(r):
            return -np.exp(-(r**2) / 4)
        
        corrected, diagnostics = apply_backreaction_correction(
            original_energy, 2.0, test_profile, quick_estimate=True
        )
        
        # Should reduce energy by ~15%
        self.assertLess(corrected, original_energy)
        self.assertAlmostEqual(corrected / original_energy, 0.85, places=2)
        self.assertEqual(diagnostics["method"], "empirical")
    
    def test_energy_reduction_bounds(self):
        """Test energy reduction stays within reasonable bounds."""
        original_energy = 1.0
        
        def test_profile(r):
            return -np.exp(-(r**2) / 4)
        
        corrected, _ = apply_backreaction_correction(
            original_energy, 2.0, test_profile, quick_estimate=True
        )
        
        # Reduction should be reasonable (not more than 50%)
        self.assertGreater(corrected, 0.5 * original_energy)
        self.assertLess(corrected, original_energy)


class TestEnhancementPathways(unittest.TestCase):
    """Test enhancement pathway calculations."""
    
    def setUp(self):
        self.config = EnhancementConfig(
            cavity_Q=1e6,
            squeezing_db=15.0,
            num_bubbles=3
        )
        
    def test_cavity_boost_calculator(self):
        """Test cavity boost enhancement calculations."""
        cavity_calc = CavityBoostCalculator(self.config)
        
        # Test enhancement factor calculation
        enhancement = cavity_calc.casimir_enhancement_factor(1e6, 1.0)
        self.assertGreater(enhancement, 1.0)
        self.assertLess(enhancement, 100.0)  # Reasonable upper bound
        
        # Higher Q should give more enhancement
        low_Q_enhancement = cavity_calc.casimir_enhancement_factor(1e4, 1.0)
        high_Q_enhancement = cavity_calc.casimir_enhancement_factor(1e8, 1.0)
        self.assertGreater(high_Q_enhancement, low_Q_enhancement)
        
    def test_quantum_squeezing_enhancer(self):
        """Test quantum squeezing enhancement calculations."""
        squeezing_calc = QuantumSqueezingEnhancer(self.config)
        
        # Test enhancement factor
        enhancement = squeezing_calc.squeezing_enhancement_factor(15.0, 0.1)
        self.assertGreater(enhancement, 1.0)
        self.assertLess(enhancement, 50.0)
        
        # Higher squeezing should give more enhancement
        low_sq = squeezing_calc.squeezing_enhancement_factor(5.0, 0.1)
        high_sq = squeezing_calc.squeezing_enhancement_factor(25.0, 0.1)
        self.assertGreater(high_sq, low_sq)
        
        # Test achievable squeezing levels
        current = squeezing_calc.estimate_achievable_squeezing("current")
        future = squeezing_calc.estimate_achievable_squeezing("near_future")
        theoretical = squeezing_calc.estimate_achievable_squeezing("theoretical")
        
        self.assertLess(current, future)
        self.assertLess(future, theoretical)
    
    def test_multi_bubble_superposition(self):
        """Test multi-bubble superposition calculations."""
        multi_bubble = MultiBubbleSuperposition(self.config)
        
        # Test interference pattern calculation
        positions = [(0, 0, 0), (2, 0, 0), (-2, 0, 0)]
        phases = [0, 0, 0]  # In-phase
        points = np.array([(x, 0, 0) for x in np.linspace(-5, 5, 20)])
        
        interference = multi_bubble.bubble_interference_pattern(positions, phases, points)
        self.assertEqual(len(interference), len(points))
        
        # Test enhancement factor
        enhancement = multi_bubble.superposition_enhancement_factor(3)
        self.assertGreater(enhancement, 1.0)
        self.assertLess(enhancement, 10.0)  # Reasonable bound
        
        # More bubbles should generally give more enhancement
        enhancement_2 = multi_bubble.superposition_enhancement_factor(2)
        enhancement_5 = multi_bubble.superposition_enhancement_factor(5)
        self.assertGreater(enhancement_5, enhancement_2)
    
    def test_enhancement_orchestrator(self):
        """Test complete enhancement pathway orchestration."""
        orchestrator = EnhancementPathwayOrchestrator(self.config)
        
        base_energy = 1.0
        results = orchestrator.combine_all_enhancements(base_energy)
        
        # Check all enhancement types are present
        self.assertIn("cavity_enhancement", results)
        self.assertIn("squeezing_enhancement", results)
        self.assertIn("multi_bubble_enhancement", results)
        self.assertIn("total_enhancement", results)
        self.assertIn("final_energy", results)
        
        # Final energy should be reduced
        self.assertLess(results["final_energy"], base_energy)
        
        # Total enhancement should be product of individual enhancements
        expected_total = (results["cavity_enhancement"] * 
                         results["squeezing_enhancement"] * 
                         results["multi_bubble_enhancement"])
        self.assertAlmostEqual(results["total_enhancement"], expected_total, places=5)


class TestEnhancementPipeline(unittest.TestCase):
    """Test the complete enhancement pipeline."""
    
    def setUp(self):
        self.config = PipelineConfig(
            grid_resolution=10,  # Small for testing
            convergence_tolerance=1e-3,
            max_iterations=20
        )
        self.pipeline = WarpBubbleEnhancementPipeline(self.config)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertIsNotNone(self.pipeline.config)
        self.assertIsNotNone(self.pipeline.enhancement_orchestrator)
        self.assertEqual(len(self.pipeline.results_history), 0)
    
    def test_base_energy_calculation(self):
        """Test base energy requirement calculation."""
        mu, R = 0.10, 2.3
        base_energy = self.pipeline.compute_base_energy_requirement(mu, R)
        
        self.assertGreater(base_energy, 0)
        self.assertIsInstance(base_energy, float)
    
    def test_apply_all_corrections(self):
        """Test application of all corrections and enhancements."""
        mu, R = 0.10, 2.3
        base_energy = 1.0
        
        corrections = self.pipeline.apply_all_corrections(base_energy, mu, R)
        
        self.assertIn("base_energy", corrections)
        self.assertIn("final_energy", corrections)
        self.assertIn("total_reduction_factor", corrections)
        
        # Final energy should be less than base energy
        self.assertLess(corrections["final_energy"], base_energy)
        
        # Reduction factor should be positive and less than 1
        reduction = corrections["total_reduction_factor"]
        self.assertGreater(reduction, 0)
        self.assertLess(reduction, 1)
    
    def test_parameter_space_scan(self):
        """Test parameter space scanning."""
        scan_results = self.pipeline.scan_parameter_space(detailed_scan=False)
        
        self.assertIn("energy_grid", scan_results)
        self.assertIn("feasibility_grid", scan_results)
        self.assertIn("best_configuration", scan_results)
        self.assertIn("feasible_configurations", scan_results)
        
        # Check grid dimensions
        resolution = self.config.grid_resolution if self.config.grid_resolution <= 20 else 20
        energy_grid = scan_results["energy_grid"]
        self.assertEqual(energy_grid.shape[0], resolution)
        self.assertEqual(energy_grid.shape[1], resolution)
    
    def test_quick_feasibility_check(self):
        """Test quick feasibility check function."""
        result = run_quick_feasibility_check()
        
        self.assertIn("parameters", result)
        self.assertIn("base_energy", result)
        self.assertIn("final_energy", result)
        self.assertIn("feasible", result)
        
        # Check parameter values
        params = result["parameters"]
        self.assertEqual(params["mu"], 0.10)
        self.assertEqual(params["R"], 2.3)
        
        # Energy should be positive
        self.assertGreater(result["base_energy"], 0)
        self.assertGreater(result["final_energy"], 0)
    
    def test_pipeline_summary_generation(self):
        """Test pipeline summary generation."""
        # Create minimal results for testing
        test_results = {
            "lqg_analysis": {
                "optimal_parameters": {
                    "mu_optimal": 0.10,
                    "R_optimal": 2.3,
                    "enhancement_factor": 2.3
                }
            },
            "parameter_scan": {
                "num_feasible": 5,
                "best_configuration": {"final_energy": 0.8},
                "unity_configurations": [{"mu": 0.1, "R": 2.3}]
            },
            "convergence_analysis": {
                "converged": True,
                "final_energy": 0.95,
                "iterations": 10
            }
        }
        
        summary = self.pipeline.generate_pipeline_summary(test_results)
        
        self.assertIn("lqg_enhancement", summary)
        self.assertIn("feasibility", summary)
        self.assertIn("convergence", summary)
        self.assertIn("overall_assessment", summary)
        
        # Check overall feasibility assessment
        self.assertTrue(summary["overall_assessment"]["warp_bubble_feasible"])


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_end_to_end_feasibility(self):
        """Test end-to-end feasibility analysis."""
        # Use optimal LQG parameters
        optimal = optimal_lqg_parameters()
        mu = optimal["mu_optimal"]
        R = optimal["R_optimal"]
        
        # Create pipeline with reasonable settings
        config = PipelineConfig(
            grid_resolution=5,  # Very small for testing
            max_iterations=5
        )
        pipeline = WarpBubbleEnhancementPipeline(config)
        
        # Compute energy requirement
        base_energy = pipeline.compute_base_energy_requirement(mu, R)
        corrections = pipeline.apply_all_corrections(base_energy, mu, R)
        
        # Check that enhancements do reduce energy
        final_energy = corrections["final_energy"]
        self.assertLess(final_energy, base_energy)
        
        # Energy should be positive and finite
        self.assertGreater(final_energy, 0)
        self.assertTrue(np.isfinite(final_energy))
    
    def test_enhancement_consistency(self):
        """Test that enhancement factors are applied consistently."""
        # Test that applying enhancements in different orders gives same result
        config = EnhancementConfig()
        
        # Individual calculators
        cavity = CavityBoostCalculator(config)
        squeezing = QuantumSqueezingEnhancer(config)
        multi_bubble = MultiBubbleSuperposition(config)
        
        # Get individual enhancement factors
        cavity_factor = cavity.casimir_enhancement_factor(config.cavity_Q, config.cavity_volume)
        squeezing_factor = squeezing.squeezing_enhancement_factor(config.squeezing_db)
        bubble_factor = multi_bubble.superposition_enhancement_factor(config.num_bubbles)
        
        # Combined enhancement
        expected_total = cavity_factor * squeezing_factor * bubble_factor
        
        # Test through orchestrator
        orchestrator = EnhancementPathwayOrchestrator(config)
        results = orchestrator.combine_all_enhancements(1.0)
        
        self.assertAlmostEqual(results["total_enhancement"], expected_total, places=5)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    # Run all tests
    unittest.main(verbosity=2)
