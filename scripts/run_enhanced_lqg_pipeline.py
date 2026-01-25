"""
Enhanced LQG Warp Bubble Analysis Runner

This script runs the complete LQG-enhanced warp bubble analysis pipeline,
integrating all recent discoveries and enhancements to achieve feasible
warp bubble configurations.

Usage:
    python run_enhanced_lqg_pipeline.py [--config CONFIG_FILE] [--output OUTPUT_FILE]
    python run_enhanced_lqg_pipeline.py --quick-check
    python run_enhanced_lqg_pipeline.py --find-unity
    python run_enhanced_lqg_pipeline.py --parameter-scan
"""

import argparse
import sys
import json
import logging
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.warp_qft.enhancement_pipeline import (
    PipelineConfig, WarpBubbleEnhancementPipeline,
    run_quick_feasibility_check, find_first_unity_configuration
)
from src.warp_qft.enhancement_pathway import EnhancementConfig
from src.warp_qft.lqg_profiles import optimal_lqg_parameters, compare_profile_types


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('lqg_pipeline.log')
        ]
    )


def load_config(config_file: str) -> PipelineConfig:
    """Load pipeline configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Create enhancement config if provided
        enhancement_config = None
        if 'enhancement_config' in config_data:
            enh_data = config_data.pop('enhancement_config')
            enhancement_config = EnhancementConfig(**enh_data)
        
        # Create main config
        config = PipelineConfig(**config_data)
        if enhancement_config:
            config.enhancement_config = enhancement_config
            
        return config
        
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Using default configuration.")
        return PipelineConfig()
    except Exception as e:
        print(f"Error loading config: {e}. Using default configuration.")
        return PipelineConfig()


def save_config_template(filename: str):
    """Save a template configuration file."""
    config = PipelineConfig()
    
    config_dict = {
        "mu_min": config.mu_min,
        "mu_max": config.mu_max,
        "R_min": config.R_min,
        "R_max": config.R_max,
        "lqg_profile": config.lqg_profile,
        "use_backreaction": config.use_backreaction,
        "backreaction_quick": config.backreaction_quick,
        "backreaction_iterative": config.backreaction_iterative,
        "backreaction_outer_iterations": config.backreaction_outer_iterations,
        "backreaction_relative_energy_tolerance": config.backreaction_relative_energy_tolerance,
        "use_cavity_boost": config.use_cavity_boost,
        "use_squeezing": config.use_squeezing,
        "use_multi_bubble": config.use_multi_bubble,
        "grid_resolution": config.grid_resolution,
        "convergence_tolerance": config.convergence_tolerance,
        "max_iterations": config.max_iterations,
        "target_energy_ratio": config.target_energy_ratio,
        "enhancement_config": {
            "cavity_Q": config.enhancement_config.cavity_Q,
            "squeezing_db": config.enhancement_config.squeezing_db,
            "num_bubbles": config.enhancement_config.num_bubbles,
            "cavity_volume": config.enhancement_config.cavity_volume,
            "squeezing_bandwidth": config.enhancement_config.squeezing_bandwidth,
            "bubble_separation": config.enhancement_config.bubble_separation
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"Configuration template saved to {filename}")


def run_quick_check():
    """Run a quick feasibility check."""
    print("=" * 60)
    print("QUICK FEASIBILITY CHECK")
    print("=" * 60)
    
    # Use optimal LQG parameters
    optimal = optimal_lqg_parameters()
    mu_opt = optimal["mu_optimal"]
    R_opt = optimal["R_optimal"]
    
    print(f"Testing optimal LQG parameters: μ={mu_opt:.3f}, R={R_opt:.2f}")
    
    result = run_quick_feasibility_check(mu_opt, R_opt)
    
    print(f"\nResults:")
    print(f"  Base energy requirement: {result['base_energy']:.4f}")
    print(f"  Final energy requirement: {result['final_energy']:.4f}")
    print(f"  Feasible (≤ unity): {result['feasible']}")
    print(f"  Energy reduction: {(1 - result['final_energy']/result['base_energy'])*100:.1f}%")
    
    # Show enhancement breakdown
    corrections = result['enhancement_breakdown']
    if 'enhancements' in corrections:
        enh = corrections['enhancements']
        print(f"\nEnhancement breakdown:")
        print(f"  Cavity boost: {enh.get('cavity_enhancement', 1.0):.2f}×")
        print(f"  Quantum squeezing: {enh.get('squeezing_enhancement', 1.0):.2f}×")
        print(f"  Multi-bubble: {enh.get('multi_bubble_enhancement', 1.0):.2f}×")
        print(f"  Total enhancement: {enh.get('total_enhancement', 1.0):.2f}×")
    
    return result


def run_unity_search():
    """Search for the first unity configuration."""
    print("=" * 60)
    print("UNITY CONFIGURATION SEARCH")
    print("=" * 60)
    
    print("Searching for parameter configuration achieving unity energy requirement...")
    
    unity_config = find_first_unity_configuration()
    
    if "error" in unity_config:
        print(f"❌ {unity_config['error']}")
        return None
    else:
        print(f"✅ Unity configuration found!")
        print(f"   μ = {unity_config['mu']:.6f}")
        print(f"   R = {unity_config['R']:.6f}")
        print(f"   Energy = {unity_config['energy']:.6f}")
        print(f"   Deviation from unity: {abs(unity_config['energy'] - 1.0):.2e}")
        
        return unity_config


def run_parameter_scan(config: PipelineConfig):
    """Run a parameter space scan."""
    print("=" * 60)
    print("PARAMETER SPACE SCAN")
    print("=" * 60)
    
    pipeline = WarpBubbleEnhancementPipeline(config)
    
    print(f"Scanning parameter space:")
    print(f"  μ range: [{config.mu_min:.3f}, {config.mu_max:.3f}]")
    print(f"  R range: [{config.R_min:.2f}, {config.R_max:.2f}]")
    print(f"  Grid resolution: {config.grid_resolution}×{config.grid_resolution}")
    
    scan_results = pipeline.scan_parameter_space(detailed_scan=True)
    
    print(f"\nScan Results:")
    print(f"  Feasible configurations found: {scan_results['num_feasible']}")
    
    if scan_results['best_configuration']:
        best = scan_results['best_configuration']
        print(f"  Best configuration:")
        print(f"    μ = {best['mu']:.6f}")
        print(f"    R = {best['R']:.6f}")
        print(f"    Energy = {best['final_energy']:.6f}")
    
    unity_configs = scan_results.get('unity_configurations', [])
    if unity_configs:
        print(f"  Unity configurations found: {len(unity_configs)}")
        best_unity = unity_configs[0]
        print(f"  Best near-unity:")
        print(f"    μ = {best_unity['mu']:.6f}")
        print(f"    R = {best_unity['R']:.6f}")
        print(f"    Energy = {best_unity['energy']:.6f}")
    
    return scan_results


def run_complete_pipeline(config: PipelineConfig, output_file: str):
    """Run the complete enhancement pipeline."""
    print("=" * 60)
    print("COMPLETE LQG ENHANCEMENT PIPELINE")
    print("=" * 60)
    
    pipeline = WarpBubbleEnhancementPipeline(config)
    
    print("Running complete pipeline analysis...")
    print("This may take several minutes depending on grid resolution.")
    
    try:
        results = pipeline.run_complete_pipeline(save_results=True, output_file=output_file)
        
        # Display summary
        summary = results.get('summary', {})
        
        print(f"\n{'=' * 40}")
        print("PIPELINE SUMMARY")
        print(f"{'=' * 40}")
        
        # LQG Enhancement
        lqg_summary = summary.get('lqg_enhancement', {})
        if lqg_summary:
            print(f"LQG Enhancement:")
            print(f"  Optimal μ: {lqg_summary.get('optimal_mu', 'N/A'):.6f}")
            print(f"  Optimal R: {lqg_summary.get('optimal_R', 'N/A'):.6f}")
            print(f"  Enhancement factor: {lqg_summary.get('max_enhancement_factor', 'N/A'):.2f}×")
        
        # Feasibility
        feasibility = summary.get('feasibility', {})
        if feasibility:
            print(f"\nFeasibility Analysis:")
            print(f"  Feasible configurations: {feasibility.get('feasible_configurations_found', 0)}")
            print(f"  Best energy achieved: {feasibility.get('best_energy_achieved', 'N/A'):.6f}")
            print(f"  Unity configurations: {feasibility.get('unity_configurations', 0)}")
        
        # Convergence
        convergence = summary.get('convergence', {})
        if convergence:
            print(f"\nConvergence Analysis:")
            print(f"  Achieved unity: {convergence.get('achieved_unity', False)}")
            print(f"  Final energy ratio: {convergence.get('final_energy_ratio', 'N/A'):.6f}")
            print(f"  Iterations required: {convergence.get('iterations_required', 'N/A')}")
        
        # Enhancement pathways
        pathways = summary.get('enhancement_pathways', {})
        if pathways:
            print(f"\nEnhancement Pathways:")
            print(f"  Required enhancement: {pathways.get('required_enhancement', 'N/A'):.2f}×")
            print(f"  Achievable enhancement: {pathways.get('achievable_enhancement', 'N/A'):.2f}×")
            print(f"  Pathway feasible: {pathways.get('pathway_feasible', False)}")
        
        # Overall assessment
        overall = summary.get('overall_assessment', {})
        if overall:
            print(f"\nOverall Assessment:")
            feasible = overall.get('warp_bubble_feasible', False)
            print(f"  Warp bubble feasible: {'✅ YES' if feasible else '❌ NO'}")
            print(f"  Primary mechanism: {overall.get('primary_enhancement_mechanism', 'Unknown')}")
            print(f"  Key breakthrough: {overall.get('key_breakthrough', 'None identified')}")
        
        print(f"\nDetailed results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        logging.exception("Complete pipeline execution failed")
        return None


def run_profile_comparison():
    """Run LQG profile comparison analysis."""
    print("=" * 60)
    print("LQG PROFILE COMPARISON")
    print("=" * 60)
    
    optimal = optimal_lqg_parameters()
    mu_opt = optimal["mu_optimal"]
    R_opt = optimal["R_optimal"]
    
    print(f"Comparing profiles at optimal parameters: μ={mu_opt:.3f}, R={R_opt:.2f}")
    
    comparison = compare_profile_types(mu_opt, R_opt)
    
    print(f"\nProfile Energy Comparison:")
    toy_energy = comparison['toy_model']
    
    for profile, energy in comparison.items():
        if profile == 'toy_model':
            print(f"  {profile:15}: {energy:.6f} (baseline)")
        else:
            enhancement = energy / toy_energy
            print(f"  {profile:15}: {energy:.6f} ({enhancement:.2f}× enhancement)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced LQG Warp Bubble Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_enhanced_lqg_pipeline.py --quick-check
  python run_enhanced_lqg_pipeline.py --find-unity
  python run_enhanced_lqg_pipeline.py --parameter-scan
  python run_enhanced_lqg_pipeline.py --complete --output results.json
  python run_enhanced_lqg_pipeline.py --config my_config.json --complete
        """
    )
    
    # Action arguments
    parser.add_argument('--quick-check', action='store_true',
                       help='Run quick feasibility check at optimal parameters')
    parser.add_argument('--find-unity', action='store_true',
                       help='Search for first unity configuration')
    parser.add_argument('--parameter-scan', action='store_true',
                       help='Run parameter space scan')
    parser.add_argument('--profile-comparison', action='store_true',
                       help='Compare LQG profile types')
    parser.add_argument('--complete', action='store_true',
                       help='Run complete pipeline analysis')
    
    # Configuration arguments
    parser.add_argument('--config', type=str,
                       help='Configuration file (JSON format)')
    parser.add_argument('--save-config-template', type=str,
                       help='Save configuration template to file')
    parser.add_argument('--output', type=str, default='lqg_pipeline_results.json',
                       help='Output file for results')
    
    # Backreaction arguments
    parser.add_argument('--backreaction-iterative', action='store_true',
                       help='Use iterative/nonlinear backreaction coupling')
    parser.add_argument('--backreaction-outer-iters', type=int, default=10,
                       help='Max outer iterations for iterative backreaction (default: 10)')
    parser.add_argument('--backreaction-rel-tol', type=float, default=1e-4,
                       help='Relative energy tolerance for iterative backreaction (default: 1e-4)')
    
    # General arguments
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Handle config template generation
    if args.save_config_template:
        save_config_template(args.save_config_template)
        return
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        config = PipelineConfig()
        print("Using default configuration")
    
    # Apply CLI backreaction overrides if provided
    if args.backreaction_iterative:
        config.backreaction_iterative = True
        config.backreaction_outer_iterations = args.backreaction_outer_iters
        config.backreaction_relative_energy_tolerance = args.backreaction_rel_tol
        print(f"Iterative backreaction enabled: max_iters={config.backreaction_outer_iterations}, rel_tol={config.backreaction_relative_energy_tolerance}")
    
    # Execute requested analysis
    if args.quick_check:
        run_quick_check()
        
    elif args.find_unity:
        run_unity_search()
        
    elif args.parameter_scan:
        run_parameter_scan(config)
        
    elif args.profile_comparison:
        run_profile_comparison()
        
    elif args.complete:
        run_complete_pipeline(config, args.output)
        
    else:
        # Default: run quick check if no action specified
        print("No specific action requested. Running quick feasibility check.")
        print("Use --help for available options.")
        print()
        run_quick_check()


if __name__ == "__main__":
    main()
