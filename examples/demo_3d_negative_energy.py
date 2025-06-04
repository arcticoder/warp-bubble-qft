#!/usr/bin/env python3
"""
3D Negative Energy Shell Analysis

This script implements the 3D spherical shell analysis for quantum inequality
violations in polymer field theory, including parameter optimization.
"""

import numpy as np
from scipy.integrate import simps
from scipy.special import sinc  # sinc(x) = sin(pi x)/(pi x) in SciPy; adjust accordingly

# ----------------------------
# 1. Quantify the Negative-Energy Shell in 3D
# ----------------------------

def sampling_function(t, tau):
    """
    Gaussian sampling function f(t) = exp(-t^2/(2 τ^2)) / (sqrt(2π) τ).
    """
    return np.exp(-t**2 / (2 * tau**2)) / (np.sqrt(2 * np.pi) * tau)


def pi_shell(r, R, sigma, A, omega, t):
    """
    Spherically symmetric π(r,t) = A * exp(- (r - R)^2 / (2 σ^2) ) * sin(ω t).
    Returns a 1D array over r.
    """
    envelope = np.exp(- ((r - R) ** 2) / (2 * sigma**2))
    return A * envelope * np.sin(omega * t)


def energy_density_polymer(pi_r, mu):
    """
    ρ(r) = ½ [ (sin(μ π(r))/μ)^2 ]   (neglecting φ gradient term for simplicity)
    pi_r: array of π(r) over radial grid
    mu: polymer scale
    Returns array of same shape.
    """
    # ordinary numpy sinc = sin(pi*x)/(pi*x), so we compute sin(μ π)/(μ) directly
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
        A = A_factor * (np.pi / (2 * mu))
        for tau in tau_values:
            for R in R_values:
                I_val = compute_I(mu, tau, R, sigma, A, omega, r_max, Nr, t_max, Nt)
                results[(mu, tau, R)] = I_val
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
    # using numpy sinc: np.sinc(x/π) = sin(x)/x, so sin(mu)/mu = np.sinc(mu/np.pi)
    sinc_mu = np.sin(mu) / mu
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
# 2,4,5. Skeletons / Placeholders for Further Steps
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
    # TODO: implement 3+1D discretization, stress-energy coupling to Einstein solver
    raise NotImplementedError("3+1D evolution is a large undertaking; implement as needed.")


def linearized_stability_analysis(phi_0, pi_0, mu, grid_shape, 
                                  dt, dx, num_steps):
    """
    Placeholder: perform linearized perturbation analysis around known solution (φ_0, π_0).
    - phi_0, pi_0: background solution arrays
    - mu: polymer scale
    Returns: growth rates or damping factors for perturbations.
    """
    # TODO: discretize δφ, δπ, evolve linearized equations, compute norm growth
    raise NotImplementedError("Implement perturbation evolution and eigenvalue analysis.")


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
    # TODO: set up PDE solver for metric functions, use T_{μν}^poly as source
    raise NotImplementedError("Implement metric solver with polymer stress-energy coupling.")


# Example parameter setup:
if __name__ == "__main__":
    print("🌌 3D Negative Energy Shell Analysis")
    print("=" * 50)
    
    # Define parameter ranges
    mu_values = [0.1, 0.3, 0.6, 1.0]
    tau_values = [0.5, 1.0, 2.0]
    R_values = [2.0, 3.0, 4.0]
    
    # Shell parameters
    sigma = 0.5                # shell thickness parameter
    A_factor = 1.1             # ensure A > π/(2μ)
    omega = np.pi / 2.5        # set T = 5.0, so ω = 2π/T = 2π/5 ≈ 1.256 (adjust as needed)
    
    # Perform scan
    print("Scanning parameter space...")
    results = scan_parameters(mu_values, tau_values, R_values, sigma, A_factor, omega)
    
    # Print negative I(mu,tau,R) values
    print("\nResults (negative values indicate QI violations):")
    for key, I_val in sorted(results.items()):
        mu, tau, R = key
        status = "✅ QI VIOLATION" if I_val < 0 else "❌ No violation"
        print(f"μ={mu:.2f}, τ={tau:.2f}, R={R:.1f} → I = {I_val:.6f} {status}")
    
    # Find optimal μ
    print("\n🔍 Optimizing polymer parameter μ...")
    best_mu, best_bound, mu_vals, bound_vals = scan_mu_for_bound()
    print(f"Optimal μ ≈ {best_mu:.3f}, bound = {best_bound:.6e}")
    
    # Summary statistics
    negative_results = {k: v for k, v in results.items() if v < 0}
    if negative_results:
        print(f"\n📊 Summary: Found {len(negative_results)} QI-violating configurations")
        strongest_violation = min(negative_results.values())
        print(f"Strongest violation: I = {strongest_violation:.6f}")
    else:
        print("\n⚠️  No QI violations found with current parameters")
