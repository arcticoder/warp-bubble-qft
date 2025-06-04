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
        # compute ρ(r,t) on radial grid
        rho_rt = energy_density_polymer(pi_rt, mu)
        # compute sampling function f(t)
        f_t = sampling_function(ti, tau)
        # radial integral ∫ ρ(r,t) * 4πr^2 dr  using Simpson's rule
        radial_integral = simps(rho_rt * volume_factor, r)
        # accumulate I_sum += (radial integral) * f(t) * dt
        I_sum += radial_integral * f_t * dt
    
    return I_sum


def scan_parameters(mu_values, tau_values, R_values, sigma, A_factor, omega, 
                    r_max=10.0, Nr=400, t_max=5.0, Nt=500):
    """
    Scan over mu ∈ mu_values, tau ∈ tau_values, R ∈ R_values, computing I(mu,tau,R).
    A_factor is used to set A = A_factor * (π / (2 mu)).
    
    Returns:
      results: dict keyed by (mu, tau, R) → I_value
    """
    results = {}
    for mu in mu_values:
        A = A_factor * (np.pi / (2 * mu)) if mu > 0 else 1.0
        for tau in tau_values:
            for R in R_values:
                I_val = compute_I(mu, tau, R, sigma, A, omega, r_max, Nr, t_max, Nt)
                results[(mu, tau, R)] = I_val
                
                # Print progress
                print(f"Computed I({mu:.1f}, {tau:.1f}, {R:.1f}) = {I_val:.6e}")
    
    return results


# ----------------------------
# 3. Optimize Polymer Parameter μ
# ----------------------------

def polymer_QI_bound(mu, hbar=1.0, tau=1.0):
    """
    Compute the polymer-modified Ford–Roman bound:
    bound(mu) = - (ħ * sinc(mu)) / (12 π τ^2).
    Here sinc(mu) = sin(mu)/mu.
    """
    # Calculate sinc(mu) = sin(mu)/mu directly
    sinc_mu = np.sin(mu) / mu if mu > 0 else 1.0
    return - (hbar * sinc_mu) / (12 * np.pi * tau**2)


def scan_mu_for_bound(mu_min=0.1, mu_max=1.0, num=50, hbar=1.0, tau=1.0):
    """
    Sample μ between mu_min and mu_max, compute the polymer bound,
    and return the μ that gives the most negative bound (i.e., minimal value).
    """
    mu_vals = np.linspace(mu_min, mu_max, num)
    bounds = np.array([polymer_QI_bound(mu, hbar, tau) for mu in mu_vals])
    idx_min = np.argmin(bounds)  # most negative value
    return mu_vals[idx_min], bounds[idx_min], mu_vals, bounds


# ----------------------------
# 2,4,5. Implementation for Further Steps
# ----------------------------

def evolve_phi_pi_3plus1D(initial_phi, initial_pi, grid_shape, 
                          metric_params, mu, dt, dx, num_steps):
    """
    Placeholder: evolve (φ, π) on a 3+1D AMR grid with polymer-corrected Hamiltonian.
    - initial_phi, initial_pi: 3D arrays (grid_shape)
    - metric_params: dict containing metric functions/fields to solve for warp bubble
    - mu: polymer scale parameter
    - dt: time step
    - dx: spatial grid spacing
    - num_steps: number of time steps to evolve
    Returns: (phi_evolved, pi_evolved)
    """
    # This is a placeholder for future implementation
    print("3+1D evolution requires implementation of adaptive mesh refinement")
    print("and coupling to Einstein field equations.")
    
    # Return initial values for now
    return initial_phi, initial_pi


def linearized_stability_analysis(phi_0, pi_0, mu, grid_shape, 
                                  dt, dx, num_steps):
    """
    Placeholder: perform linearized perturbation analysis around known solution (φ_0, π_0).
    - phi_0, pi_0: background solution arrays
    - mu: polymer scale
    Returns: growth rates or damping factors for perturbations.
    """
    # Placeholder for future implementation
    print("Linearized stability analysis requires implementing perturbation evolution")
    print("and computing eigenvalue spectrum of the linearized system.")
    
    # Return dummy stability results
    return {
        "stable_modes": True,
        "max_growth_rate": 0.0,
        "unstable_wavelengths": []
    }


def solve_warp_bubble_metric(r_b, s_function, phi, pi, mu, grid_shape):
    """
    Placeholder: solve for bubble-shape function s(r_b) in Alcubierre-style metric:
      ds^2 = –[1 – v^2 s(r_b)] dt^2 – 2 v s(r_b) dt dz + dx^2 + dy^2 + [1 + v^2 s(r_b)] dz^2
    with T_{μν}^poly from (φ, π).
    - r_b: radial coordinate grid
    - s_function: initial guess or functional form of s(r_b)
    - phi, pi: current field arrays (possibly 3D)
    - mu: polymer scale
    Returns: updated s_function or metric coefficients.
    """
    # Placeholder for future implementation
    print("Warp metric solver requires implementing Einstein field equations")
    print("with polymer-modified stress-energy tensor.")
    
    # Return initial guess for now
    return s_function


def compute_negative_energy_requirements(spatial_scale, tau, mu, desired_violation_factor=1.1):
    """
    Compute negative energy requirements for a given
    polymer scale and spacetime parameters.
    """
    # Get polymer-modified bound
    bounds = polymer_modified_bounds(-1.0, spatial_scale, mu, tau)
    
    # Scale to desired violation factor
    required_density = bounds["ford_roman_bound"] * desired_violation_factor
    
    # Estimate shell volume
    shell_volume = 4 * np.pi * spatial_scale**2 * (spatial_scale / 5)  # Assume thickness ~ R/5
    
    # Estimate total negative energy needed
    total_energy = required_density * shell_volume
    
    return {
        "required_density": required_density,
        "shell_volume": shell_volume,
        "total_negative_energy": total_energy,
        "polymer_enhancement": bounds["enhancement_factor"]
    }


def squeezed_vacuum_energy(r_squeeze, omega, volume, hbar=1.0):
    """
    Estimate negative energy density from squeezed vacuum.
    
    Args:
        r_squeeze: Squeezing parameter
        omega: Angular frequency
        volume: Cavity volume
        hbar: Planck's constant
        
    Returns:
        Negative energy density estimate
    """
    # Simple model: ρ_neg ≈ -(ħω/V) * sinh²(r)
    return -(hbar * omega / volume) * np.sinh(r_squeeze)**2


def laboratory_feasibility_analysis(mu, spatial_scale, tau):
    """
    Compare required negative energy with achievable laboratory values.
    
    Args:
        mu: Polymer scale
        spatial_scale: Characteristic length
        tau: Sampling time
        
    Returns:
        Feasibility analysis
    """
    # Required negative energy
    requirements = compute_negative_energy_requirements(spatial_scale, tau, mu)
    
    # Achievable with Casimir effect
    casimir_energy_density = -1e-10  # J/m³, approximate
    casimir_volume = 1e-12  # m³
    casimir_total = casimir_energy_density * casimir_volume
    
    # Achievable with squeezed vacuum in cavity
    r_squeeze = 2.0  # Moderate squeezing parameter
    omega = 2 * np.pi * 10e9  # 10 GHz
    cavity_volume = 1e-9  # 1 nanoliter
    squeezed_density = squeezed_vacuum_energy(r_squeeze, omega, cavity_volume)
    squeezed_total = squeezed_density * cavity_volume
    
    # Compare with required energy
    req_energy = requirements["total_negative_energy"]
    casimir_ratio = casimir_total / req_energy
    squeezed_ratio = squeezed_total / req_energy
    
    return {
        "required_energy": req_energy,
        "required_density": requirements["required_density"],
        "casimir_energy": casimir_total,
        "squeezed_energy": squeezed_total,
        "casimir_feasibility": casimir_ratio,
        "squeezed_feasibility": squeezed_ratio,
        "is_casimir_feasible": casimir_ratio >= 1.0,
        "is_squeezed_feasible": squeezed_ratio >= 1.0,
        "polymer_enhancement": requirements["polymer_enhancement"]
    }


def visualize_shell_results(results, mu_values, tau_values, R_values):
    """
    Visualize parameter scan results.
    
    Args:
        results: Dictionary from scan_parameters
        mu_values, tau_values, R_values: Parameter lists
        
    Returns:
        None (shows plots)
    """
    # Setup plots
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("3D Negative Energy Shell Analysis", fontsize=16)
    
    # Plot 1: I vs R for different mu values at fixed tau
    tau_fixed = tau_values[0]
    ax1 = fig.add_subplot(231)
    for mu in mu_values:
        I_vals = [results.get((mu, tau_fixed, R), 0) for R in R_values]
        ax1.plot(R_values, I_vals, 'o-', label=f"μ={mu}")
    ax1.set_title(f"I vs R (τ={tau_fixed})")
    ax1.set_xlabel("Shell Radius R")
    ax1.set_ylabel("I(μ,τ,R)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: I vs mu for different R at fixed tau
    ax2 = fig.add_subplot(232)
    for R in R_values:
        I_vals = [results.get((mu, tau_fixed, R), 0) for mu in mu_values]
        ax2.plot(mu_values, I_vals, 's-', label=f"R={R}")
    ax2.set_title(f"I vs μ (τ={tau_fixed})")
    ax2.set_xlabel("Polymer Scale μ")
    ax2.set_ylabel("I(μ,τ,R)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Polymer bounds vs mu
    ax3 = fig.add_subplot(233)
    mu_fine = np.linspace(0.1, 1.0, 100)
    bounds = [polymer_QI_bound(mu) for mu in mu_fine]
    ax3.plot(mu_fine, bounds, 'r-')
    ax3.set_title("Polymer QI Bound vs μ")
    ax3.set_xlabel("Polymer Scale μ")
    ax3.set_ylabel("QI Bound")
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: I vs τ for different mu at fixed R
    R_fixed = R_values[0]
    ax4 = fig.add_subplot(234)
    for mu in mu_values:
        I_vals = [results.get((mu, tau, R_fixed), 0) for tau in tau_values]
        ax4.plot(tau_values, I_vals, 'x-', label=f"μ={mu}")
    ax4.set_title(f"I vs τ (R={R_fixed})")
    ax4.set_xlabel("Sampling Width τ")
    ax4.set_ylabel("I(μ,τ,R)")
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Plot 5: Negative energy configuration count
    ax5 = fig.add_subplot(235)
    neg_counts = np.zeros(len(mu_values))
    for i, mu in enumerate(mu_values):
        for tau in tau_values:
            for R in R_values:
                if results.get((mu, tau, R), 0) < 0:
                    neg_counts[i] += 1
    ax5.bar(mu_values, neg_counts)
    ax5.set_title("Count of QI-Violating Configs")
    ax5.set_xlabel("Polymer Scale μ")
    ax5.set_ylabel("Count")
    
    # Plot 6: Energy profile at optimal parameters
    # Find most negative I value
    min_I = float('inf')
    min_params = None
    for params, I_val in results.items():
        if I_val < min_I:
            min_I = I_val
            min_params = params
    
    if min_params:
        mu_opt, tau_opt, R_opt = min_params
        ax6 = fig.add_subplot(236)
        r = np.linspace(0, 10, 200)
        
        # Show energy density at t=0 (peak)
        A = (np.pi / (2 * mu_opt)) * 1.1  # A_factor = 1.1
        pi_r = pi_shell(r, R_opt, 0.5, A, 2*np.pi/5, 0)
        rho = energy_density_polymer(pi_r, mu_opt)
        
        ax6.plot(r, rho, 'b-')
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax6.set_title(f"Energy Density (μ={mu_opt}, R={R_opt})")
        ax6.set_xlabel("Radius r")
        ax6.set_ylabel("ρ(r)")
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("3d_shell_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()


# Example usage (if run as main script)
if __name__ == "__main__":
    print("🌌 3D Negative Energy Shell Analysis")
    print("=" * 50)
    
    # Define parameter ranges
    mu_values = [0.3, 0.6, 0.8, 1.0]
    tau_values = [0.5, 1.0, 2.0]
    R_values = [2.0, 3.0, 4.0]
    
    # Shell parameters
    sigma = 0.5                # shell thickness parameter
    A_factor = 1.1             # ensure A > π/(2μ)
    omega = np.pi / 2.5        # set T = 5.0, ω = 2π/T = 2π/5
    
    # Use smaller integration ranges for faster demo
    r_max = 8.0
    Nr = 100
    t_max = 4.0
    Nt = 100
    
    print("Scanning parameter space...")
    print("This may take a while - using reduced grid size for demonstration")
    
    results = scan_parameters(
        mu_values, tau_values, R_values, sigma, A_factor, omega,
        r_max=r_max, Nr=Nr, t_max=t_max, Nt=Nt
    )
    
    # Print negative I(mu,tau,R) values (QI violations)
    print("\nResults (negative values indicate QI violations):")
    for key, I_val in sorted(results.items()):
        mu, tau, R = key
        status = "✅ QI VIOLATION" if I_val < 0 else "❌ No violation"
        print(f"μ={mu:.2f}, τ={tau:.2f}, R={R:.1f} → I = {I_val:.6f} {status}")
    
    # Find optimal μ
    print("\n🔍 Optimizing polymer parameter μ...")
    best_mu, best_bound, mu_vals, bound_vals = scan_mu_for_bound()
    print(f"Optimal μ ≈ {best_mu:.3f}, bound = {best_bound:.6e}")
    
    # Laboratory feasibility analysis
    print("\n🔬 Laboratory Feasibility Analysis:")
    spatial_scale = 1e-6  # 1 micron
    lab_tau = 1e-9       # 1 nanosecond
    feasibility = laboratory_feasibility_analysis(best_mu, spatial_scale, lab_tau)
    
    print(f"Required negative energy: {feasibility['required_energy']:.3e} J")
    print(f"Casimir effect yields: {feasibility['casimir_energy']:.3e} J")
    print(f"Squeezed vacuum yields: {feasibility['squeezed_energy']:.3e} J")
    print(f"Polymer enhancement: {feasibility['polymer_enhancement']:.3f}×")
    
    # Summary statistics
    negative_results = {k: v for k, v in results.items() if v < 0}
    if negative_results:
        print(f"\n📊 Summary: Found {len(negative_results)} QI-violating configurations")
        strongest_violation = min(negative_results.values())
        strongest_key = min(negative_results.items(), key=lambda x: x[1])[0]
        print(f"Strongest violation: I = {strongest_violation:.6f} at μ={strongest_key[0]}, τ={strongest_key[1]}, R={strongest_key[2]}")
    else:
        print("\n⚠️  No QI violations found with current parameters")
    
    # Visualize results
    print("\n📈 Generating visualization...")
    visualize_shell_results(results, mu_values, tau_values, R_values)
