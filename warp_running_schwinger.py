#!/usr/bin/env python3
"""
Running Coupling Implementation for Warp Bubble QFT

This module implements the complete β-function running coupling α_eff(E) 
and embeds it directly into the Schwinger pair production formula.

α_eff(E) = α₀ / (1 - (b/(2π))α₀ ln(E/E₀))
Γ_Sch^poly = (α_eff eE)² / (4π³ℏc) exp[-πm²c³/(eEℏ) F(μ_g)]
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class RunningCouplingConfig:
    """Configuration for running coupling calculations."""
    alpha_0: float = 1/137.036  # Fine structure constant
    E_0: float = 0.511e-3      # Reference energy (electron mass in GeV)
    e_charge: float = 1.602e-19  # Elementary charge
    hbar: float = 1.055e-34     # Reduced Planck constant
    c: float = 3e8              # Speed of light
    m_electron: float = 0.511e-3  # Electron mass in GeV
    mu_g: float = 0.15          # Polymer parameter

class WarpBubbleRunningSchwinger:
    """
    Complete running coupling implementation for warp bubble QFT calculations.
    
    This class provides the ACTUAL Schwinger rates with β-function corrections
    that must be used in ALL warp drive energy analysis.
    """
    
    def __init__(self, config: RunningCouplingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def beta_function_coefficient(self, n_fermions: int = 2) -> float:
        """
        Compute the β-function coefficient b = (11N_c - 2N_f)/3 for QED.
        For QED: b = -2N_f/3 where N_f is number of fermions.
        """
        return -2 * n_fermions / 3
    
    def running_coupling(self, E: float, b_coeff: float) -> float:
        """
        THE running coupling formula α_eff(E) with β-function corrections.
        
        This is the KEY formula that must replace all instances of α₀
        in Schwinger calculations.
        
        Formula: α_eff(E) = α₀ / (1 - (b/(2π))α₀ ln(E/E₀))
        """
        if E <= 0:
            raise ValueError("Energy must be positive")
        
        ln_ratio = np.log(E / self.config.E_0)
        beta_factor = b_coeff / (2 * np.pi)
        denominator = 1 - beta_factor * self.config.alpha_0 * ln_ratio
        
        # Avoid singularity
        if abs(denominator) < 1e-12:
            self.logger.warning(f"Running coupling near Landau pole at E={E}")
            return self.config.alpha_0 * 1e6  # Large but finite value
        
        return self.config.alpha_0 / denominator
    
    def polymer_suppression_factor(self, E: float) -> float:
        """
        Polymer suppression F(μ_g) = sin²(μ_g E)/(μ_g E)².
        
        This modifies the Schwinger exponential in LQG.
        """
        mu_g_E = self.config.mu_g * E
        if abs(mu_g_E) < 1e-12:
            return 1.0  # limₓ→₀ sin²(x)/x² = 1
        
        sinc_factor = np.sin(mu_g_E) / mu_g_E
        return sinc_factor**2
    def schwinger_rate_with_running_coupling(self, E: float, b_coeff: float) -> float:
        """
        THE complete Schwinger production rate with running coupling and polymer corrections.
        
        This is the production rate that MUST be used in all warp drive calculations.
        
        Formula: Γ_Sch^poly = (α_eff eE)² / (4π³ℏc) exp[-πm²c³/(eEℏ) F(μ_g)]
        """
        if E <= 0:
            raise ValueError("Electric field must be positive")
        
        # Get running coupling
        alpha_eff = self.running_coupling(E, b_coeff)
        
        # Polymer suppression
        F_polymer = self.polymer_suppression_factor(E)
        
        # Convert to natural units for calculation
        E_natural = E * 1e9 * 1.602e-19 / (1.973e-16)  # Convert GeV to natural units
        
        # Schwinger rate components
        prefactor = (alpha_eff * E_natural)**2 / (4 * np.pi**3)
        
        # Exponential with polymer correction  
        m_electron_natural = self.config.m_electron * 1e9 * 1.602e-19 / (1.973e-16)
        exponent_classical = -np.pi * m_electron_natural**2 / E_natural
        exponent_polymer = exponent_classical * F_polymer
        
        # Ensure exponent isn't too negative
        if exponent_polymer < -100:
            rate = 0.0
        else:
            rate = prefactor * np.exp(exponent_polymer)
        
        return rate
    
    def generate_rate_vs_field_curves(self, E_range: Tuple[float, float], 
                                    b_values: List[float], n_points: int = 100) -> Dict:
        """
        Generate rate-vs-field curves for different β-function coefficients.
        
        This produces the REQUIRED curves for b = {0, 5, 10}.
        """
        print(f"🔷 Generating Schwinger rate curves for b = {b_values}...")
        
        E_min, E_max = E_range
        E_grid = np.logspace(np.log10(E_min), np.log10(E_max), n_points)
        
        results = {
            'config': {
                'alpha_0': self.config.alpha_0,
                'E_0': self.config.E_0,
                'mu_g': self.config.mu_g,
                'm_electron': self.config.m_electron
            },
            'E_grid': E_grid.tolist(),
            'b_values': b_values,
            'rates': {},
            'classical_rates': [],
            'enhancement_factors': {}
        }
        
        # Classical rates (b=0, no running)
        classical_rates = []
        for E in E_grid:
            rate_classical = self.schwinger_rate_with_running_coupling(E, 0.0)
            classical_rates.append(rate_classical)
        
        results['classical_rates'] = classical_rates
        
        # Rates for each b value
        for b in b_values:
            rates_b = []
            enhancements_b = []
            
            for i, E in enumerate(E_grid):
                rate_b = self.schwinger_rate_with_running_coupling(E, b)
                enhancement = rate_b / classical_rates[i] if classical_rates[i] > 0 else 1.0
                
                rates_b.append(rate_b)
                enhancements_b.append(enhancement)
            
            results['rates'][f'b_{b}'] = rates_b
            results['enhancement_factors'][f'b_{b}'] = enhancements_b
        
        # Generate plot
        plt.figure(figsize=(12, 8))
        
        # Plot rates
        plt.subplot(2, 1, 1)
        for b in b_values:
            rates = results['rates'][f'b_{b}']
            plt.loglog(E_grid, rates, label=f'b = {b}', linewidth=2)
        
        plt.loglog(E_grid, classical_rates, 'k--', label='Classical (b=0)', linewidth=1)
        plt.xlabel('Electric Field E [GeV]')
        plt.ylabel('Schwinger Rate Γ [s⁻¹]')
        plt.title('Schwinger Production Rates with Running Coupling')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot enhancement factors
        plt.subplot(2, 1, 2)
        for b in b_values:
            if b != 0:  # Skip b=0 since it's the reference
                enhancements = results['enhancement_factors'][f'b_{b}']
                plt.semilogx(E_grid, enhancements, label=f'b = {b}', linewidth=2)
        
        plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        plt.xlabel('Electric Field E [GeV]')
        plt.ylabel('Rate Enhancement Factor')
        plt.title('Running Coupling Enhancement over Classical')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = "schwinger_running_coupling_curves.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Rate-vs-field curves saved to: {plot_file}")
        
        return results
    
    def validate_classical_limit(self, E_test: float = 1e-6, tolerance: float = 1e-10) -> bool:
        """
        Verify that b→0 recovers classical Schwinger formula.
        """
        rate_classical = self.schwinger_rate_with_running_coupling(E_test, 0.0)
        
        # Expected classical rate without running
        alpha_0 = self.config.alpha_0
        prefactor_expected = (alpha_0 * self.config.e_charge * E_test)**2 / (
            4 * np.pi**3 * self.config.hbar * self.config.c
        )
        
        # Should match to high precision for b=0
        relative_error = abs(rate_classical - prefactor_expected * np.exp(
            -np.pi * (self.config.m_electron * self.config.c**2)**2 / (
                self.config.e_charge * E_test * self.config.hbar * self.config.c
            ) * self.polymer_suppression_factor(E_test)
        )) / rate_classical
        
        return relative_error < tolerance
    
    def export_schwinger_data(self, output_file: str, E_range: Tuple[float, float] = (1e-6, 1e-3),
                             b_values: List[float] = [0, 5, 10]) -> Dict:
        """
        Export complete Schwinger data with running coupling for integration.
        """
        print(f"🔷 Exporting Schwinger data to {output_file}...")
        
        # Generate curves
        results = self.generate_rate_vs_field_curves(E_range, b_values)
        
        # Add validation
        validation_passed = self.validate_classical_limit()
        results['validation'] = {
            'classical_limit_check': validation_passed
        }
        
        # Add summary statistics
        summary = {}
        for b in b_values:
            rates = results['rates'][f'b_{b}']
            summary[f'b_{b}'] = {
                'min_rate': min(rates),
                'max_rate': max(rates),
                'mean_rate': np.mean(rates),
                'rate_span_orders_of_magnitude': np.log10(max(rates) / min(rates))
            }
        
        results['summary'] = summary
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"   Schwinger data exported with {len(b_values)} β-function cases")
        print(f"   Field range: {E_range[0]:.2e} - {E_range[1]:.2e} GeV")
        print(f"   Validation: {'PASSED' if validation_passed else 'FAILED'}")
        
        return results

# Integration function for the main warp bubble pipeline
def integrate_running_schwinger_into_warp_pipeline() -> bool:
    """
    MAIN INTEGRATION FUNCTION: Embed running coupling Schwinger rates into warp bubble calculations.
    
    This function modifies the warp bubble pipeline to use running coupling
    in all pair production rate calculations.
    """
    print("🔷 Integrating Running Coupling into Warp Bubble Pipeline...")
    
    # Initialize running coupling calculator
    config = RunningCouplingConfig(
        alpha_0=1/137.036,
        mu_g=0.15,
        E_0=0.511e-3
    )
    schwinger = WarpBubbleRunningSchwinger(config)    # Generate the required rate-vs-field curves for b = {0, 5, 10}
    E_range = (1e-2, 1e0)  # Higher field range for visible rates
    b_values = [0, 5, 10]   # β-function coefficients as specified
    
    # Export complete data
    output_file = "warp_bubble_running_schwinger_integration.json"
    results = schwinger.export_schwinger_data(output_file, E_range, b_values)
    
    # Validate integration
    validation_passed = results['validation']['classical_limit_check']
    
    if validation_passed:
        print("✅ Running coupling Schwinger rates successfully integrated")
        
        # Create marker file for downstream processes
        with open("RUNNING_SCHWINGER_INTEGRATED.flag", 'w') as f:
            f.write(f"Running Schwinger rates integrated with b_values={b_values}")
        
        # Print summary
        print(f"   Generated curves for b = {b_values}")
        print(f"   Field range: {E_range[0]:.2e} - {E_range[1]:.2e} GeV")
        for b in b_values:
            summary = results['summary'][f'b_{b}']
            print(f"   b={b}: rate span {summary['rate_span_orders_of_magnitude']:.1f} orders of magnitude")
    else:
        print("❌ Running coupling integration failed validation")
    
    return validation_passed

if __name__ == "__main__":
    # Test the running coupling implementation
    config = RunningCouplingConfig(mu_g=0.15)
    schwinger = WarpBubbleRunningSchwinger(config)
    
    # Test parameters
    E_test = 1e-4  # GeV
    b_test = 5.0
    
    # Compute rates
    alpha_eff = schwinger.running_coupling(E_test, b_test)
    rate_running = schwinger.schwinger_rate_with_running_coupling(E_test, b_test)
    rate_classical = schwinger.schwinger_rate_with_running_coupling(E_test, 0.0)
    
    print(f"Running coupling: α_eff({E_test:.2e} GeV) = {alpha_eff:.6e}")
    print(f"Classical α₀ = {config.alpha_0:.6e}")
    print(f"Enhancement factor: {alpha_eff/config.alpha_0:.3f}")
    print(f"")
    print(f"Schwinger rates:")
    print(f"  Running (b={b_test}): {rate_running:.6e} s⁻¹")
    print(f"  Classical (b=0): {rate_classical:.6e} s⁻¹") 
    print(f"  Rate enhancement: {rate_running/rate_classical:.3f}")
    
    # Run validation
    validation_ok = schwinger.validate_classical_limit()
    print(f"")
    print(f"Classical limit validation: {'✓' if validation_ok else '✗'}")
