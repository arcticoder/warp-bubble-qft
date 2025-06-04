#!/usr/bin/env python3
"""
3D Negative Energy Shell Analysis

This script implements the 3D spherical shell analysis for quantum inequality
violations in polymer field theory, including parameter optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.special import sinc  # sinc(x) = sin(pi x)/(pi x) in SciPy; adjust accordingly
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from warp_qft.negative_energy import sampling_function, ford_roman_violation_analysis
from warp_qft.stability import polymer_modified_bounds, violation_duration

# ----------------------------
# 1. Quantify the Negative-Energy Shell in 3D
# ----------------------------

def pi_shell(r, R, sigma, A, omega, t):
    """
    Spherically symmetric π(r,t) = A * exp(- (r - R)^2 / (2 σ^2) ) * sin(ω t).
    Returns a 1D array over r.
    
    Args:
        r: radial coordinate array
        R: shell radius
        sigma: shell thickness
        A: amplitude
        omega: frequency
        t: time
        
    Returns:
        Array of π values over radial grid
    """
    envelope = np.exp(- ((r - R) ** 2) / (2 * sigma**2))
    return A * envelope * np.sin(omega * t)


def energy_density_polymer(pi_r, mu):
    """
    ρ(r) = ½ [ (sin(μ π(r))/μ)^2 ]   (neglecting φ gradient term for simplicity)
    
    Args:
        pi_r: array of π(r) over radial grid
        mu: polymer scale
        
    Returns:
        Array of energy density over radial grid
    """
    if mu == 0:
        # Classical case
        return 0.5 * pi_r**2
        
    # Polymer case: ρ = (1/2) * [sin(μ*π)/μ]^2
    return 0.5 * (np.sin(mu * pi_r) / mu) ** 2


def compute_I(mu, tau, R, sigma, A, omega, 
              r_max=10.0, Nr=400, 
              t_max=5.0, Nt=500):
    """
    Compute I(mu) = ∫_{r=0}^∞ ∫_{t=-T/2}^{T/2} ρ_eff(r,t) f(t) 4π r^2 dr dt
    using discrete grids in r and t.
    
    Parameters:
      mu: polymer scale
      tau: sampling width
      R: shell radius
      sigma: shell thickness parameter
      A: amplitude (must satisfy A > π/(2 μ))
      omega: temporal frequency
      r_max: maximum radius to integrate (assume outside shell it decays)
      Nr: number of radial sample points
      t_max: half-duration for time integration
      Nt: number of time sample points
    
    Returns:
      I_value: float, approximate value of I(mu,tau,R).
    """
    # radial grid from 0 to r_max
    r = np.linspace(0, r_max, Nr)
    dr = r[1] - r[0]
    
    # time grid from -t_max to t_max
    t = np.linspace(-t_max, t_max, Nt)
    dt = t[1] - t[0]
    
    # 4π r^2 factor for volume element
    volume_factor = 4 * np.pi * r**2
    
    I_sum = 0.0
    for ti in t:
        # compute π(r, t) over radial grid
        pi_rt = pi_shell(r, R, sigma, A, omega, ti)
        
        # compute energy density ρ_eff(r, t)
        rho_rt = energy_density_polymer(pi_rt, mu)
        
        # compute sampling function f(t)
        f_t = sampling_function(ti, tau)
        
        # radial integral: ∫ ρ_eff(r,t) * 4π r^2 dr
        radial_integrand = rho_rt * volume_factor
        radial_integral = simps(radial_integrand, r)
        
        # accumulate in time integral
        I_sum += radial_integral * f_t * dt
    
    return I_sum


def scan_parameters(mu_values, tau_values, R_values, sigma, A_factor, omega, 
                    r_max=10.0, Nr=400, t_max=5.0, Nt=500):
    """
    Scan over (μ, τ, R) parameter space and compute I(μ, τ, R) for each combination.
    
    Args:
        mu_values: list/array of polymer scale values
        tau_values: list/array of sampling width values  
        R_values: list/array of shell radius values
        sigma: shell thickness parameter
        A_factor: amplitude factor (A = A_factor * π/(2μ))
        omega: temporal frequency
        r_max, Nr, t_max, Nt: integration parameters
        
    Returns:
        results: dict mapping (mu, tau, R) -> I_value
        violations: dict tracking which combinations violate Ford-Roman bound
    """
    results = {}
    violations = {}
    
    total_combos = len(mu_values) * len(tau_values) * len(R_values)
    combo_count = 0
    
    print(f"Scanning {total_combos} parameter combinations...")
    
    for mu in mu_values:
        # Amplitude must satisfy A > π/(2μ) to enter negative energy regime
        A = A_factor * (np.pi / (2 * mu))
        
        for tau in tau_values:
            for R in R_values:
                combo_count += 1
                
                print(f"Progress: {combo_count}/{total_combos} - "
                      f"μ={mu:.2f}, τ={tau:.2f}, R={R:.2f}", end='\r')
                
                # Compute the integral I(μ, τ, R)
                I_value = compute_I(mu, tau, R, sigma, A, omega, 
                                  r_max, Nr, t_max, Nt)
                results[(mu, tau, R)] = I_value
                
                # Check if this violates the polymer Ford-Roman bound
                polymer_bound = polymer_QI_bound(mu, tau=tau)
                violates_bound = I_value < polymer_bound
                violations[(mu, tau, R)] = violates_bound
    
    print()  # newline after progress
    return results, violations


# ----------------------------
# 3. Optimize Polymer Parameter μ
# ----------------------------

def polymer_QI_bound(mu, hbar=1.0, tau=1.0):
    """
    Compute the polymer-modified Ford-Roman bound:
    Bound = - (ħ * sinc(μ)) / (12 π τ^2)
    where sinc(μ) = sin(μ)/μ.
    
    Args:
        mu: polymer scale parameter
        hbar: reduced Planck constant (default 1.0)
        tau: sampling function width
        
    Returns:
        Polymer Ford-Roman bound value (negative)
    """
    if mu == 0:
        # Classical limit: sinc(0) = 1
        sinc_mu = 1.0
    else:
        sinc_mu = np.sin(mu) / mu
    
    return - (hbar * sinc_mu) / (12 * np.pi * tau**2)


def scan_mu_for_bound(mu_min=0.1, mu_max=1.0, num=50, hbar=1.0, tau=1.0):
    """
    Scan μ values to find the most relaxed (most negative) polymer QI bound.
    
    Args:
        mu_min, mu_max: range of μ values
        num: number of μ values to sample
        hbar, tau: physical parameters
        
    Returns:
        best_mu: μ value giving most relaxed bound
        best_bound: corresponding bound value
        mu_vals: array of all μ values tested
        bound_vals: array of corresponding bound values
    """
    mu_vals = np.linspace(mu_min, mu_max, num)
    bound_vals = np.array([polymer_QI_bound(mu, hbar, tau) for mu in mu_vals])
    
    # Most relaxed bound is the most negative (minimum value)
    idx_min = np.argmin(bound_vals)
    best_mu = mu_vals[idx_min]
    best_bound = bound_vals[idx_min]
    
    return best_mu, best_bound, mu_vals, bound_vals


# ----------------------------
# 2,4,5. Implementation for Further Steps
# ----------------------------

def evolve_phi_pi_3plus1D(initial_phi, initial_pi, grid_shape, 
                          metric_params, mu, dt, dx, num_steps):
    """
    Placeholder: 3+1D evolution of (φ, π) on AMR grid with polymer Hamiltonian.
    
    Would implement:
    ∂φ/∂t = sin(μπ)/μ
    ∂π/∂t = ∇²φ - m²φ + metric coupling terms
    
    Args:
        initial_phi, initial_pi: initial field configurations
        grid_shape: (Nx, Ny, Nz) grid dimensions
        metric_params: parameters for Alcubierre metric
        mu: polymer scale
        dt, dx: time and spatial steps
        num_steps: number of evolution steps
        
    Returns:
        phi_final, pi_final: evolved field configurations
        metric_evolution: time series of metric functions
    """
    print("⚠️  evolve_phi_pi_3plus1D: Placeholder implementation")
    print("    Real implementation would:")
    print("    - Set up 3D AMR grid with adaptive refinement")
    print("    - Evolve polymer Hamiltonian equations")
    print("    - Couple to Alcubierre metric solver")
    print("    - Track energy conservation and stability")
    
    return initial_phi, initial_pi, {}


def linearized_stability_analysis(phi_0, pi_0, mu, grid_shape, 
                                  dt, dx, num_steps):
    """
    Placeholder: Linearized stability analysis around background (φ₀, π₀).
    
    Would implement:
    δ̇φ = cos(μπ₀) δπ
    δ̇π = ∇²δφ - m²δφ
    
    Args:
        phi_0, pi_0: background field configuration
        mu: polymer scale
        grid_shape: spatial grid dimensions
        dt, dx: temporal and spatial steps
        num_steps: evolution steps for perturbation analysis
        
    Returns:
        eigenvalues: spectrum of linearized operator
        eigenvectors: corresponding mode shapes
        stability_verdict: stable/unstable classification
    """
    print("⚠️  linearized_stability_analysis: Placeholder implementation")
    print("    Real implementation would:")
    print("    - Linearize polymer field equations around background")
    print("    - Compute eigenvalue spectrum of linearized operator")
    print("    - Check for tachyonic or growing modes")
    print("    - Verify no superluminal propagation")
    
    return [], [], "stable"


def solve_warp_bubble_metric(r_b, s_function, phi, pi, mu, grid_shape):
    """
    Placeholder: Solve for Alcubierre metric given polymer stress-energy tensor.
    
    Would solve Einstein equations:
    R_μν - ½gR = 8πG T_μν^polymer
    
    with T_μν^polymer containing sin(μπ)/μ modifications.
    
    Args:
        r_b: bubble radius
        s_function: initial guess for shape function
        phi, pi: polymer field configuration
        mu: polymer scale
        grid_shape: spatial discretization
        
    Returns:
        metric_components: (g_tt, g_rr, g_θθ, g_φφ)
        shape_function: optimized s(r)
        energy_conditions: analysis of energy condition violations
    """
    print("⚠️  solve_warp_bubble_metric: Placeholder implementation")
    print("    Real implementation would:")
    print("    - Compute polymer stress-energy tensor T_μν^poly")
    print("    - Solve Einstein field equations with polymer source")
    print("    - Optimize Alcubierre shape function s(r)")
    print("    - Verify energy condition violations are controlled")
    
    return {}, lambda r: np.exp(-r**2), {}


def compute_negative_energy_requirements(spatial_scale, tau, mu, desired_violation_factor=1.1):
    """
    Compute the negative energy density required to violate Ford-Roman bound by a given factor.
    
    Args:
        spatial_scale: characteristic bubble size
        tau: sampling function width
        mu: polymer scale
        desired_violation_factor: how much to violate bound (>1.0)
        
    Returns:
        required_rho_neg: negative energy density needed
        total_energy_deficit: total negative energy in bubble
    """
    # Polymer Ford-Roman bound
    bound = polymer_QI_bound(mu, tau=tau)
    
    # Target integral value (more negative than bound)
    target_I = bound * desired_violation_factor
    
    # Estimate required energy density (rough approximation)
    # Assume energy concentrated in sphere of radius ~ spatial_scale
    volume_estimate = (4/3) * np.pi * spatial_scale**3
    
    # Required energy density to achieve target I
    required_rho_neg = target_I / volume_estimate
    total_energy_deficit = required_rho_neg * volume_estimate
    
    return required_rho_neg, total_energy_deficit


def squeezed_vacuum_energy(r_squeeze, omega, volume, hbar=1.0):
    """
    Estimate negative energy density from squeezed vacuum state.
    
    For a squeezed state with parameter r_squeeze:
    ρ_neg ≈ - (ħω/V) sinh(r_squeeze)
    
    Args:
        r_squeeze: squeezing parameter (dimensionless)
        omega: characteristic frequency
        volume: cavity volume
        hbar: reduced Planck constant
        
    Returns:
        rho_neg: negative energy density
    """
    return - (hbar * omega / volume) * np.sinh(r_squeeze)


def laboratory_feasibility_analysis(mu, spatial_scale, tau):
    """
    Analyze experimental feasibility of polymer warp bubble.
    
    Compares required negative energy against achievable sources:
    - Casimir effect
    - Squeezed vacuum states
    - Dynamic Casimir effect
    
    Args:
        mu: polymer scale
        spatial_scale: bubble size (meters)
        tau: observation timescale (seconds)
        
    Returns:
        feasibility_report: dictionary with energy requirements and sources
    """
    # Required negative energy for violation
    rho_req, E_req = compute_negative_energy_requirements(spatial_scale, tau, mu)
    
    # Casimir energy estimate (parallel plates)
    hbar_c = 1.055e-34  # J⋅s
    c = 3e8  # m/s
    L = spatial_scale
    E_casimir = - (np.pi**2 * hbar_c * c) / (240 * L**4)  # J/m³
    E_casimir_total = E_casimir * (spatial_scale**3)
    
    # Squeezed vacuum energy (microwave cavity at 5 GHz, r_squeeze = 1)
    omega = 2 * np.pi * 5e9  # rad/s
    cavity_volume = (spatial_scale)**3  # m³
    r_squeeze = 1.0  # achievable squeezing
    E_squeezed = squeezed_vacuum_energy(r_squeeze, omega, cavity_volume, hbar_c)
    E_squeezed_total = E_squeezed * cavity_volume
    
    # Feasibility ratios
    casimir_ratio = abs(E_casimir_total / E_req) if E_req != 0 else 0
    squeezed_ratio = abs(E_squeezed_total / E_req) if E_req != 0 else 0
    
    return {
        "required_energy_density": rho_req,
        "required_total_energy": E_req,
        "casimir_energy_density": E_casimir,
        "casimir_total_energy": E_casimir_total,
        "squeezed_energy_density": E_squeezed,
        "squeezed_total_energy": E_squeezed_total,
        "casimir_feasibility_ratio": casimir_ratio,
        "squeezed_feasibility_ratio": squeezed_ratio,
        "spatial_scale": spatial_scale,
        "polymer_scale": mu,
        "observation_time": tau,
        "feasible_with_casimir": casimir_ratio >= 1.0,
        "feasible_with_squeezed": squeezed_ratio >= 1.0
    }


def visualize_shell_results(results, mu_values, tau_values, R_values):
    """
    Create comprehensive visualization of 3D shell analysis results.
    
    Six-panel plot showing:
    1. I vs R for different μ
    2. I vs μ for different τ  
    3. Polymer QI bound vs μ
    4. I vs τ for different R
    5. Count of violating configurations
    6. Energy density profile at optimal parameters
    
    Args:
        results: dict from scan_parameters
        mu_values, tau_values, R_values: parameter arrays
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('3D Negative Energy Shell Analysis', fontsize=16)
    
    # Panel 1: I vs R for different μ (fixed τ)
    ax1 = axes[0, 0]
    tau_fixed = tau_values[len(tau_values)//2]  # middle τ value
    for mu in mu_values:
        I_vals = [results.get((mu, tau_fixed, R), 0) for R in R_values]
        ax1.plot(R_values, I_vals, 'o-', label=f'μ={mu:.2f}')
    ax1.set_xlabel('Shell Radius R')
    ax1.set_ylabel('Integral I')
    ax1.set_title(f'I vs R (τ={tau_fixed:.2f})')
    ax1.legend()
    ax1.grid(True)
    
    # Panel 2: I vs μ for different τ (fixed R)
    ax2 = axes[0, 1]
    R_fixed = R_values[len(R_values)//2]  # middle R value
    for tau in tau_values:
        I_vals = [results.get((mu, tau, R_fixed), 0) for mu in mu_values]
        ax2.plot(mu_values, I_vals, 's-', label=f'τ={tau:.2f}')
    ax2.set_xlabel('Polymer Scale μ')
    ax2.set_ylabel('Integral I')
    ax2.set_title(f'I vs μ (R={R_fixed:.2f})')
    ax2.legend()
    ax2.grid(True)
    
    # Panel 3: Polymer QI bound vs μ
    ax3 = axes[0, 2]
    tau_for_bound = tau_values[0]  # use first τ value
    bound_vals = [polymer_QI_bound(mu, tau=tau_for_bound) for mu in mu_values]
    ax3.plot(mu_values, bound_vals, 'r-', linewidth=2, label='Polymer QI Bound')
    ax3.set_xlabel('Polymer Scale μ')
    ax3.set_ylabel('QI Bound')
    ax3.set_title(f'QI Bound vs μ (τ={tau_for_bound:.2f})')
    ax3.grid(True)
    ax3.legend()
    
    # Panel 4: I vs τ for different R (fixed μ)
    ax4 = axes[1, 0]
    mu_fixed = mu_values[len(mu_values)//2]  # middle μ value
    for R in R_values:
        I_vals = [results.get((mu_fixed, tau, R), 0) for tau in tau_values]
        ax4.plot(tau_values, I_vals, '^-', label=f'R={R:.1f}')
    ax4.set_xlabel('Sampling Width τ')
    ax4.set_ylabel('Integral I')
    ax4.set_title(f'I vs τ (μ={mu_fixed:.2f})')
    ax4.legend()
    ax4.grid(True)
    
    # Panel 5: Count of violating configurations
    ax5 = axes[1, 1]
    violation_counts = []
    for mu in mu_values:
        count = 0
        for tau in tau_values:
            for R in R_values:
                I_val = results.get((mu, tau, R), 0)
                bound = polymer_QI_bound(mu, tau=tau)
                if I_val < bound:
                    count += 1
        violation_counts.append(count)
    
    ax5.bar(range(len(mu_values)), violation_counts, 
            tick_label=[f'{mu:.2f}' for mu in mu_values])
    ax5.set_xlabel('Polymer Scale μ')
    ax5.set_ylabel('# Violating Configs')
    ax5.set_title('Ford-Roman Violations by μ')
    ax5.grid(True, axis='y')
    
    # Panel 6: Energy density profile at optimal parameters
    ax6 = axes[1, 2]
    # Find the parameter combination giving most negative I
    min_I = float('inf')
    best_params = None
    for key, I_val in results.items():
        if I_val < min_I:
            min_I = I_val
            best_params = key
    
    if best_params:
        mu_opt, tau_opt, R_opt = best_params
        A_opt = 1.2 * (np.pi / (2 * mu_opt))  # Use A_factor = 1.2
        omega_opt = 2 * np.pi  # Standard frequency
        sigma = 0.5  # Shell thickness
        
        # Plot energy density profile
        r_vals = np.linspace(0, 8, 200)
        t_snapshot = 0  # t = 0
        pi_profile = pi_shell(r_vals, R_opt, sigma, A_opt, omega_opt, t_snapshot)
        rho_profile = energy_density_polymer(pi_profile, mu_opt)
        
        ax6.plot(r_vals, rho_profile, 'g-', linewidth=2)
        ax6.axvline(R_opt, color='red', linestyle='--', alpha=0.7, label=f'R={R_opt:.1f}')
        ax6.set_xlabel('Radius r')
        ax6.set_ylabel('Energy Density ρ')
        ax6.set_title(f'ρ(r) at Optimal μ={mu_opt:.2f}')
        ax6.grid(True)
        ax6.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig


# Example usage (if run as main script)
if __name__ == "__main__":
    print("🌌 3D Negative Energy Shell Analysis")
    print("=" * 50)
    
    # Parameters for the shell analysis
    mu_values = [0.1, 0.3, 0.6, 1.0]
    tau_values = [0.5, 1.0, 2.0]
    R_values = [2.0, 3.0, 4.0]
    
    # Fixed parameters
    sigma = 0.5        # shell thickness
    A_factor = 1.2     # amplitude factor: A = A_factor * π/(2μ)
    omega = 2 * np.pi  # temporal frequency
    
    print(f"μ values: {mu_values}")
    print(f"τ values: {tau_values}")
    print(f"R values: {R_values}")
    
    # Run parameter scan
    print("\n🔍 Running 3D parameter scan...")
    results, violations = scan_parameters(
        mu_values, tau_values, R_values, 
        sigma, A_factor, omega,
        r_max=10.0, Nr=300, t_max=4.0, Nt=400
    )
    
    # Display results for some key combinations
    print("\n📊 Key Results:")
    print("μ     τ     R     I(μ,τ,R)     Status")
    print("-" * 45)
    
    for mu in [0.3, 0.6]:
        for tau in [1.0]:
            for R in [2.0, 3.0]:
                key = (mu, tau, R)
                I_val = results.get(key, 0)
                bound = polymer_QI_bound(mu, tau=tau)
                status = "✅ QI VIOLATION" if I_val < bound else "❌ No violation"
                print(f"{mu:.1f}   {tau:.1f}   {R:.1f}   {I_val:+.6f}     {status}")
    
    # Find optimal μ
    print("\n🔍 Optimizing polymer parameter μ...")
    best_mu, best_bound, mu_vals, bound_vals = scan_mu_for_bound()
    print(f"Optimal μ ≈ {best_mu:.3f}, bound = {best_bound:.6e}")
    
    # Laboratory feasibility analysis
    print("\n🔬 Laboratory Feasibility Analysis:")
    spatial_scale = 1e-6  # 1 micron
    lab_tau = 1e-9       # 1 nanosecond
    feasibility = laboratory_feasibility_analysis(best_mu, spatial_scale, lab_tau)
    
    print(f"Required energy density: {feasibility['required_energy_density']:.3e} J/m³")
    print(f"Required total energy: {feasibility['required_total_energy']:.3e} J")
    print(f"Casimir energy density: {feasibility['casimir_energy_density']:.3e} J/m³")
    print(f"Casimir total energy: {feasibility['casimir_total_energy']:.3e} J")
    print(f"Squeezed energy density: {feasibility['squeezed_energy_density']:.3e} J/m³")
    print(f"Squeezed total energy: {feasibility['squeezed_total_energy']:.3e} J")
    print(f"Casimir feasibility ratio: {feasibility['casimir_feasibility_ratio']:.3f}")
    print(f"Squeezed feasibility ratio: {feasibility['squeezed_feasibility_ratio']:.3f}")
    
    # Summary statistics
    negative_results = {k: v for k, v in results.items() if v < 0}
    violation_results = {k: v for k, v in results.items() if violations.get(k, False)}
    
    if violation_results:
        print(f"\n📊 Summary: Found {len(violation_results)} QI-violating configurations")
        strongest_violation = min(violation_results.values())
        strongest_key = min(violation_results.items(), key=lambda x: x[1])[0]
        print(f"Strongest violation: I = {strongest_violation:.6f} at μ={strongest_key[0]:.1f}, τ={strongest_key[1]:.1f}, R={strongest_key[2]:.1f}")
    else:
        print("\n⚠️  No QI violations found with current parameters")
    
    if negative_results:
        print(f"Found {len(negative_results)} configurations with I < 0")
    
    # Visualize results
    print("\n📈 Generating visualization...")
    visualize_shell_results(results, mu_values, tau_values, R_values)
    
    print("\n✅ 3D Analysis Complete!")
