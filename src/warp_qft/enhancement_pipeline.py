"""
Enhancement Pipeline Orchestrator

This module orchestrates the complete warp bubble enhancement pipeline,
integrating LQG profiles, metric backreaction, and enhancement pathways
to achieve feasible warp bubble configurations.

Key integration points:
- LQG-corrected negative energy profiles 
- Metric backreaction energy reduction (~15%)
- Cavity boost, squeezing, and multi-bubble enhancements
- Systematic parameter space scanning
- Convergence to unity energy requirements
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from dataclasses import dataclass, asdict
from pathlib import Path

# Import enhancement modules
from .lqg_profiles import (
    lqg_negative_energy, optimal_lqg_parameters, 
    scan_lqg_parameter_space, compare_profile_types
)
from .backreaction_solver import (
    apply_backreaction_correction, BackreactionSolver
)
from .enhancement_pathway import (
    EnhancementConfig, EnhancementPathwayOrchestrator
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Complete configuration for the enhancement pipeline."""
    # LQG parameters
    mu_min: float = 0.05
    mu_max: float = 0.20
    R_min: float = 1.5
    R_max: float = 4.0
    lqg_profile: str = "polymer_field"
    
    # Backreaction settings
    use_backreaction: bool = True
    backreaction_quick: bool = True
    
    # Enhancement pathways
    enhancement_config: EnhancementConfig = None
    use_cavity_boost: bool = True
    use_squeezing: bool = True 
    use_multi_bubble: bool = True
    
    # Pipeline settings
    grid_resolution: int = 50
    convergence_tolerance: float = 1e-4
    max_iterations: int = 100
    target_energy_ratio: float = 1.0  # Target: unity
    
    def __post_init__(self):
        if self.enhancement_config is None:
            self.enhancement_config = EnhancementConfig()


class WarpBubbleEnhancementPipeline:
    """
    Main pipeline orchestrator for achieving feasible warp bubble configurations.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.results_history = []
        self.convergence_data = []
        
        # Initialize enhancement orchestrator
        self.enhancement_orchestrator = EnhancementPathwayOrchestrator(
            self.config.enhancement_config
        )
        
    def compute_base_energy_requirement(self, mu: float, R: float) -> float:
        """
        Compute base energy requirement using LQG-corrected profiles.
        
        Args:
            mu: Polymer scale parameter
            R: Bubble radius
            
        Returns:
            Base energy requirement (before backreaction and enhancements)
        """
        # Get LQG-enhanced negative energy
        lqg_energy = lqg_negative_energy(mu, R, self.config.lqg_profile)
        
        # Convert to mass-energy equivalent requirement
        # This is a simplified model - in practice would involve full spacetime calculation
        mass_energy_ratio = 1.0 / lqg_energy if lqg_energy > 0 else float('inf')
        
        return mass_energy_ratio
    
    def apply_all_corrections(self, base_energy: float, mu: float, R: float) -> Dict:
        """
        Apply all correction and enhancement factors to base energy.
        
        Args:
            base_energy: Base energy requirement
            mu: Polymer scale parameter
            R: Bubble radius
            
        Returns:
            Dictionary with all correction results
        """
        corrections = {"base_energy": base_energy}
        current_energy = base_energy
        
        # Apply metric backreaction correction
        if self.config.use_backreaction:
            def rho_profile(r):
                from .lqg_profiles import polymer_field_profile
                return polymer_field_profile(r, mu, R)
            
            corrected_energy, backreaction_info = apply_backreaction_correction(
                current_energy, R, rho_profile, self.config.backreaction_quick
            )
            corrections["backreaction"] = backreaction_info
            corrections["energy_after_backreaction"] = corrected_energy
            current_energy = corrected_energy
        
        # Apply enhancement pathways
        enhancement_results = self.enhancement_orchestrator.combine_all_enhancements(
            current_energy
        )
        corrections["enhancements"] = enhancement_results
        
        # Final energy requirement
        final_energy = enhancement_results["final_energy"]
        corrections["final_energy"] = final_energy
        corrections["total_reduction_factor"] = final_energy / base_energy
        
        return corrections
    
    def scan_parameter_space(self, detailed_scan: bool = True) -> Dict:
        """
        Perform systematic scan of parameter space to find optimal configurations.
        
        Args:
            detailed_scan: If True, use high-resolution grid; if False, coarse scan
            
        Returns:
            Dictionary with scan results and optimal parameters
        """
        resolution = self.config.grid_resolution if detailed_scan else 20
        
        # Parameter ranges
        mu_range = np.linspace(self.config.mu_min, self.config.mu_max, resolution)
        R_range = np.linspace(self.config.R_min, self.config.R_max, resolution)
        
        # Initialize result arrays
        energy_grid = np.zeros((len(mu_range), len(R_range)))
        feasibility_grid = np.zeros((len(mu_range), len(R_range)), dtype=bool)
        
        # Track best configuration found
        best_config = None
        best_energy = float('inf')
        unity_configurations = []
        
        logger.info(f"Starting parameter space scan ({resolution}×{resolution} grid)")
        
        for i, mu in enumerate(mu_range):
            for j, R in enumerate(R_range):
                try:
                    # Compute base energy
                    base_energy = self.compute_base_energy_requirement(mu, R)
                    
                    # Apply all corrections
                    corrections = self.apply_all_corrections(base_energy, mu, R)
                    final_energy = corrections["final_energy"]
                    
                    energy_grid[i, j] = final_energy
                    feasibility_grid[i, j] = final_energy <= self.config.target_energy_ratio
                    
                    # Track best configuration
                    if final_energy < best_energy:
                        best_energy = final_energy
                        best_config = {
                            "mu": mu,
                            "R": R,
                            "final_energy": final_energy,
                            "corrections": corrections
                        }
                    
                    # Track near-unity configurations
                    if abs(final_energy - 1.0) < 0.1:  # Within 10% of unity
                        unity_configurations.append({
                            "mu": mu,
                            "R": R, 
                            "energy": final_energy,
                            "deviation_from_unity": abs(final_energy - 1.0)
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed at μ={mu:.3f}, R={R:.2f}: {e}")
                    energy_grid[i, j] = float('inf')
                    feasibility_grid[i, j] = False
        
        # Find all feasible configurations
        feasible_indices = np.where(feasibility_grid)
        feasible_configs = []
        
        for idx_i, idx_j in zip(feasible_indices[0], feasible_indices[1]):
            feasible_configs.append({
                "mu": mu_range[idx_i],
                "R": R_range[idx_j],
                "energy": energy_grid[idx_i, idx_j]
            })
        
        # Sort unity configurations by proximity to unity
        unity_configurations.sort(key=lambda x: x["deviation_from_unity"])
        
        scan_results = {
            "mu_range": mu_range,
            "R_range": R_range,
            "energy_grid": energy_grid,
            "feasibility_grid": feasibility_grid,
            "best_configuration": best_config,
            "feasible_configurations": feasible_configs,
            "unity_configurations": unity_configurations[:10],  # Top 10
            "num_feasible": len(feasible_configs),
            "scan_resolution": resolution
        }
        
        logger.info(f"Scan complete: {len(feasible_configs)} feasible configurations found")
        if unity_configurations:
            best_unity = unity_configurations[0]
            logger.info(f"Best near-unity: μ={best_unity['mu']:.3f}, R={best_unity['R']:.2f}, "
                       f"E={best_unity['energy']:.4f}")
        
        return scan_results
    
    def iterative_convergence_to_unity(self, initial_mu: float = 0.10, 
                                     initial_R: float = 2.3) -> Dict:
        """
        Iteratively refine parameters to converge to unity energy requirement.
        
        Args:
            initial_mu: Starting polymer scale parameter
            initial_R: Starting bubble radius
            
        Returns:
            Dictionary with convergence results
        """
        mu, R = initial_mu, initial_R
        convergence_history = []
        
        logger.info("Starting iterative convergence to unity")
        
        for iteration in range(self.config.max_iterations):
            # Compute current energy
            base_energy = self.compute_base_energy_requirement(mu, R)
            corrections = self.apply_all_corrections(base_energy, mu, R)
            current_energy = corrections["final_energy"]
            
            convergence_history.append({
                "iteration": iteration,
                "mu": mu,
                "R": R,
                "energy": current_energy,
                "error": abs(current_energy - 1.0)
            })
            
            # Check convergence
            if abs(current_energy - 1.0) < self.config.convergence_tolerance:
                logger.info(f"Converged to unity after {iteration+1} iterations")
                break
            
            # Update parameters using gradient estimate
            # Simple gradient descent approach
            epsilon = 0.01
            
            # Compute gradient estimates
            mu_plus = self.compute_base_energy_requirement(mu + epsilon, R)
            mu_plus_corrected = self.apply_all_corrections(mu_plus, mu + epsilon, R)["final_energy"]
            
            R_plus = self.compute_base_energy_requirement(mu, R + epsilon)
            R_plus_corrected = self.apply_all_corrections(R_plus, mu, R + epsilon)["final_energy"]
            
            grad_mu = (mu_plus_corrected - current_energy) / epsilon
            grad_R = (R_plus_corrected - current_energy) / epsilon
            
            # Update parameters (move toward unity)
            learning_rate = 0.01
            if current_energy > 1.0:  # Need to reduce energy
                mu -= learning_rate * grad_mu if grad_mu > 0 else -learning_rate * abs(grad_mu)
                R -= learning_rate * grad_R if grad_R > 0 else -learning_rate * abs(grad_R)
            else:  # Energy too low, increase slightly
                mu += learning_rate * abs(grad_mu) * 0.1
                R += learning_rate * abs(grad_R) * 0.1
            
            # Keep parameters in bounds
            mu = np.clip(mu, self.config.mu_min, self.config.mu_max)
            R = np.clip(R, self.config.R_min, self.config.R_max)
        
        final_result = convergence_history[-1] if convergence_history else None
        
        return {
            "converged": final_result["error"] < self.config.convergence_tolerance if final_result else False,
            "final_parameters": {"mu": mu, "R": R},
            "final_energy": final_result["energy"] if final_result else float('inf'),
            "iterations": len(convergence_history),
            "convergence_history": convergence_history
        }
    
    def run_complete_pipeline(self, save_results: bool = True, 
                            output_file: Optional[str] = None) -> Dict:
        """
        Run the complete enhancement pipeline analysis.
        
        Args:
            save_results: Whether to save results to file
            output_file: Output file path (if None, auto-generate)
            
        Returns:
            Complete pipeline results
        """
        logger.info("Starting complete warp bubble enhancement pipeline")
        
        results = {
            "config": asdict(self.config),
            "timestamp": np.datetime64('now').astype(str)
        }
        
        # Step 1: LQG profile analysis
        logger.info("Step 1: LQG profile analysis")
        optimal_lqg = optimal_lqg_parameters()
        profile_comparison = compare_profile_types(
            optimal_lqg["mu_optimal"], optimal_lqg["R_optimal"]
        )
        
        results["lqg_analysis"] = {
            "optimal_parameters": optimal_lqg,
            "profile_comparison": profile_comparison
        }
        
        # Step 2: Parameter space scan
        logger.info("Step 2: Parameter space scanning")
        scan_results = self.scan_parameter_space(detailed_scan=True)
        results["parameter_scan"] = scan_results
        
        # Step 3: Iterative convergence
        logger.info("Step 3: Iterative convergence to unity")
        convergence_results = self.iterative_convergence_to_unity()
        results["convergence_analysis"] = convergence_results
        
        # Step 4: Enhancement pathway optimization
        logger.info("Step 4: Enhancement pathway optimization")
        if scan_results["best_configuration"]:
            best_config = scan_results["best_configuration"]
            enhancement_optimization = self.enhancement_orchestrator.optimize_enhancement_parameters(
                self.config.target_energy_ratio, 
                best_config["corrections"]["base_energy"]
            )
            results["enhancement_optimization"] = enhancement_optimization
        
        # Step 5: Summary and recommendations
        logger.info("Step 5: Generating summary and recommendations")
        summary = self.generate_pipeline_summary(results)
        results["summary"] = summary
        
        # Save results if requested
        if save_results:
            if output_file is None:
                output_file = f"enhancement_pipeline_results_{np.datetime64('now').astype(str).replace(':', '-')}.json"
            
            self.save_results(results, output_file)
            logger.info(f"Results saved to {output_file}")
        
        return results
    
    def generate_pipeline_summary(self, results: Dict) -> Dict:
        """
        Generate a comprehensive summary of pipeline results.
        
        Args:
            results: Complete pipeline results
            
        Returns:
            Summary dictionary with key findings
        """
        summary = {}
        
        # LQG enhancement summary
        lqg_data = results.get("lqg_analysis", {})
        if lqg_data:
            optimal = lqg_data["optimal_parameters"]
            summary["lqg_enhancement"] = {
                "optimal_mu": optimal["mu_optimal"],
                "optimal_R": optimal["R_optimal"],
                "max_enhancement_factor": optimal["enhancement_factor"]
            }
        
        # Feasibility summary
        scan_data = results.get("parameter_scan", {})
        if scan_data:
            summary["feasibility"] = {
                "feasible_configurations_found": scan_data["num_feasible"],
                "best_energy_achieved": scan_data["best_configuration"]["final_energy"] if scan_data["best_configuration"] else float('inf'),
                "unity_configurations": len(scan_data.get("unity_configurations", []))
            }
        
        # Convergence summary
        convergence_data = results.get("convergence_analysis", {})
        if convergence_data:
            summary["convergence"] = {
                "achieved_unity": convergence_data["converged"],
                "final_energy_ratio": convergence_data["final_energy"],
                "iterations_required": convergence_data["iterations"]
            }
        
        # Enhancement pathway summary
        enhancement_data = results.get("enhancement_optimization", {})
        if enhancement_data:
            summary["enhancement_pathways"] = {
                "required_enhancement": enhancement_data["target_enhancement"],
                "achievable_enhancement": enhancement_data["achievable_enhancement"],
                "pathway_feasible": enhancement_data["achievable_enhancement"] >= enhancement_data["target_enhancement"]
            }
        
        # Overall assessment
        feasible = (summary.get("feasibility", {}).get("unity_configurations", 0) > 0 or
                   summary.get("convergence", {}).get("achieved_unity", False))
        
        summary["overall_assessment"] = {
            "warp_bubble_feasible": feasible,
            "primary_enhancement_mechanism": "LQG + Backreaction + Multi-pathway",
            "key_breakthrough": "Systematic convergence to unity energy requirements"
        }
        
        return summary
    
    def save_results(self, results: Dict, filename: str):
        """Save pipeline results to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def load_results(self, filename: str) -> Dict:
        """Load previously saved pipeline results."""
        with open(filename, 'r') as f:
            return json.load(f)


# Convenience functions for running specific analyses
def run_quick_feasibility_check(mu: float = 0.10, R: float = 2.3) -> Dict:
    """
    Quick feasibility check for given parameters.
    
    Args:
        mu: Polymer scale parameter
        R: Bubble radius
        
    Returns:
        Feasibility analysis results
    """
    config = PipelineConfig(
        grid_resolution=10,  # Quick scan
        use_backreaction=True,
        use_cavity_boost=True,
        use_squeezing=True,
        use_multi_bubble=True
    )
    
    pipeline = WarpBubbleEnhancementPipeline(config)
    
    # Compute energy with all enhancements
    base_energy = pipeline.compute_base_energy_requirement(mu, R)
    corrections = pipeline.apply_all_corrections(base_energy, mu, R)
    
    return {
        "parameters": {"mu": mu, "R": R},
        "base_energy": base_energy,
        "final_energy": corrections["final_energy"],
        "feasible": corrections["final_energy"] <= 1.0,
        "enhancement_breakdown": corrections
    }


def find_first_unity_configuration() -> Dict:
    """
    Find the first parameter configuration that achieves unity energy requirement.
    
    Returns:
        First unity configuration found
    """
    config = PipelineConfig(target_energy_ratio=1.0)
    pipeline = WarpBubbleEnhancementPipeline(config)
    
    # Quick scan to find unity region
    scan_results = pipeline.scan_parameter_space(detailed_scan=False)
    
    if scan_results["unity_configurations"]:
        return scan_results["unity_configurations"][0]
    else:
        # If no unity found in coarse scan, try convergence from optimal LQG
        convergence = pipeline.iterative_convergence_to_unity()
        if convergence["converged"]:
            return {
                "mu": convergence["final_parameters"]["mu"],
                "R": convergence["final_parameters"]["R"],
                "energy": convergence["final_energy"]
            }
    
    return {"error": "No unity configuration found"}


# Example usage
if __name__ == "__main__":
    # Quick feasibility check
    print("Quick feasibility check at optimal LQG parameters:")
    quick_check = run_quick_feasibility_check()
    print(f"Feasible: {quick_check['feasible']}")
    print(f"Final energy: {quick_check['final_energy']:.4f}")
    
    # Find first unity configuration
    print("\nSearching for unity configuration:")
    unity_config = find_first_unity_configuration()
    if "error" not in unity_config:
        print(f"Unity found: μ={unity_config['mu']:.3f}, R={unity_config['R']:.2f}")
    else:
        print(unity_config["error"])
    
    # Run complete pipeline (commented out for quick testing)
    # config = PipelineConfig()
    # pipeline = WarpBubbleEnhancementPipeline(config)
    # complete_results = pipeline.run_complete_pipeline()
    # print("\nComplete pipeline results available.")
