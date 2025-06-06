"""
Comprehensive Warp Bubble Analysis Module

This module provides a complete framework for analyzing warp bubble feasibility
using polymer quantum field theory, including:

1. Squeezed-vacuum negative energy estimation
2. 3D shell scanning with Ford-Roman bound checks
3. Polymer quantum inequality optimization
4. Energy requirement vs availability comparison
5. Visualization and analysis tools
6. Placeholders for full 3+1D evolution and stability

Author: Advanced Warp Bubble Research Team
Date: June 2025
"""

import numpy as np
try:
    from scipy.integrate import simpson as simps
except ImportError:
    from scipy.integrate import simps
import matplotlib.pyplot as plt

# ------------------------------------------
# 1. SQUEEZED-VACUUM NEGATIVE-ENERGY ESTIMATE
# ------------------------------------------

def squeezed_vacuum_energy(r_squeeze, omega, volume, hbar=1.055e-34):
    """
    Estimate the maximum negative energy density (J/m³) from a squeezed-vacuum state.
    Model: ρ_neg ≈ - (ħ * ω / volume) * sinh(r_squeeze).
    
    Args:
        r_squeeze (float): Squeezing parameter
        omega (float): Frequency (rad/s)
        volume (float): Volume (m³)
        hbar (float): Reduced Planck constant
        
    Returns:
        float: Negative energy density in J/m³
    """
    return - (hbar * omega / volume) * np.sinh(r_squeeze)


# ------------------------------------------
# 2. 3D NEGATIVE-ENERGY SHELL SCAN
# ------------------------------------------

def sampling_function(t, tau):
    """
    Gaussian sampling: f(t) = exp(-t^2/(2 τ^2)) / (sqrt(2π) τ).
    
    Args:
        t (float or array): Time coordinate
        tau (float): Sampling time scale
        
    Returns:
        float or array: Sampling function values
    """
    return np.exp(-t**2 / (2 * tau**2)) / (np.sqrt(2 * np.pi) * tau)

def pi_shell(r, R, sigma, A, omega, t):
    """
    π(r,t) = A * exp(- (r - R)^2 / (2 σ^2)) * sin(ω t).
    
    Args:
        r (array): Radial coordinates
        R (float): Shell radius
        sigma (float): Shell width
        A (float): Amplitude
        omega (float): Frequency
        t (float): Time
        
    Returns:
        array: π field values
    """
    return A * np.exp(- ((r - R)**2) / (2 * sigma**2)) * np.sin(omega * t)

def energy_density_polymer(pi_r, mu):
    """
    ρ_eff(r) = ½ [ (sin(π μ π(r))/(π μ))^2 ].
    
    Args:
        pi_r (array): π field values
        mu (float): Polymer scale parameter
        
    Returns:
        array: Effective energy density
    """
    return 0.5 * (np.sin(np.pi * mu * pi_r) / (np.pi * mu))**2

def polymer_QI_bound(mu, tau=1.0, hbar=1.055e-34):
    """
    Polymer-modified Ford–Roman bound:
      Bound(μ,τ) = - (ħ * sin(πμ)/(πμ)) / (12 π τ^2).
      
    Args:
        mu (float): Polymer scale parameter
        tau (float): Sampling time scale
        hbar (float): Reduced Planck constant
        
    Returns:
        float: Quantum inequality bound
    """
    sinc_mu = 1.0 if mu == 0 else np.sin(np.pi * mu)/(np.pi * mu)
    return - (hbar * sinc_mu) / (12 * np.pi * tau**2)

def compute_I_3d(mu, tau, R, sigma, A, omega, 
                 r_max=10.0, Nr=300, t_max=5.0, Nt=400):
    """
    Compute I(μ,τ,R) = ∫_{r=0}^∞ ∫_{t=-T/2}^{T/2} ρ_eff(r,t) f(t) 4π r² dr dt.
    
    Args:
        mu (float): Polymer scale parameter
        tau (float): Sampling time scale
        R (float): Shell radius
        sigma (float): Shell width
        A (float): Amplitude
        omega (float): Frequency
        r_max (float): Maximum radius for integration
        Nr (int): Number of radial points
        t_max (float): Maximum time for integration
        Nt (int): Number of time points
        
    Returns:
        float: Approximation of I integral
    """
    r = np.linspace(0, r_max, Nr)
    dr = r[1] - r[0]
    t = np.linspace(-t_max, t_max, Nt)
    dt = t[1] - t[0]
    vol_factor = 4 * np.pi * r**2

    I_sum = 0.0
    for ti in t:
        pi_rt = pi_shell(r, R, sigma, A, omega, ti)
        rho_rt = energy_density_polymer(pi_rt, mu)
        f_t = sampling_function(ti, tau)
        radial_int = simps(rho_rt * vol_factor, r)
        I_sum += radial_int * f_t * dt

    return I_sum

def scan_3d_shell(mu_vals, tau_vals, R_vals, sigma, A_factor, omega):
    """
    Scan μ ∈ mu_vals, τ ∈ tau_vals, R ∈ R_vals → compute I(μ,τ,R).
    
    Args:
        mu_vals (list): Polymer scale parameters to scan
        tau_vals (list): Sampling time scales to scan
        R_vals (list): Shell radii to scan
        sigma (float): Shell width
        A_factor (float): Amplitude factor
        omega (float): Frequency
        
    Returns:
        tuple: (results, violations) dictionaries
            results[(μ,τ,R)] = I_value
            violations[(μ,τ,R)] = True if I_value < polymer_QI_bound(μ,τ)
    """
    results    = {}
    violations = {}
    total = len(mu_vals)*len(tau_vals)*len(R_vals)
    count = 0

    for mu in mu_vals:
        A = A_factor * (np.pi/(2*mu))  # ensure μπ > π/2
        for tau in tau_vals:
            for R in R_vals:
                count += 1
                print(f"Scanning {count}/{total}: μ={mu:.2f}, τ={tau:.2f}, R={R:.2f}", end="\r")
                I_val = compute_I_3d(mu, tau, R, sigma, A, omega)
                results[(mu, tau, R)] = I_val
                violations[(mu, tau, R)] = (I_val < polymer_QI_bound(mu, tau))
    print()  # newline after scan
    return results, violations

# ------------------------------------------
# 3. μ OPTIMIZATION FOR QI BOUND
# ------------------------------------------

def find_optimal_mu(mu_min=0.1, mu_max=1.0, steps=50, tau=1.0):
    """
    Sample μ in [mu_min, mu_max] to find the most relaxed (most negative) QI bound.
    
    Args:
        mu_min (float): Minimum μ value
        mu_max (float): Maximum μ value
        steps (int): Number of steps
        tau (float): Sampling time scale
        
    Returns:
        tuple: (best_mu, best_bound, mu_array, bound_array)
    """
    mu_array = np.linspace(mu_min, mu_max, steps)
    bound_array = np.array([polymer_QI_bound(mu, tau) for mu in mu_array])
    idx = np.argmin(bound_array)
    return mu_array[idx], bound_array[idx], mu_array, bound_array

# ------------------------------------------
# 4. COMPARE REQUIRED VS. AVAILABLE NEGATIVE ENERGY
# ------------------------------------------

def required_negative_energy(mu, tau=1.0, R=3.0, dR=0.5, hbar=1.055e-34):
    """
    Rough estimate: E_req ≈ |Bound(μ,τ)| * (4π R² dR).
    
    Args:
        mu (float): Polymer scale parameter
        tau (float): Sampling time scale
        R (float): Shell radius
        dR (float): Shell thickness
        hbar (float): Reduced Planck constant
        
    Returns:
        float: Required negative energy in J
    """
    bound = polymer_QI_bound(mu, tau, hbar)
    shell_vol = 4 * np.pi * R**2 * dR
    return abs(bound) * shell_vol

def compare_neg_energy(mu, tau, R, dR, r_squeeze, omega, cavity_vol):
    """
    Compute (E_req, E_squeezed) for given parameters:
      E_req = required negative energy (J)
      E_squeezed = achievable by squeezed vacuum (J)
      
    Args:
        mu (float): Polymer scale parameter
        tau (float): Sampling time scale
        R (float): Shell radius
        dR (float): Shell thickness
        r_squeeze (float): Squeezing parameter
        omega (float): Frequency
        cavity_vol (float): Cavity volume
        
    Returns:
        tuple: (E_req, E_squeeze) in Joules
    """
    E_req = required_negative_energy(mu, tau, R, dR)
    ρ_sq = squeezed_vacuum_energy(r_squeeze, omega, cavity_vol)
    E_squeeze = ρ_sq * cavity_vol
    return E_req, E_squeeze

# ------------------------------------------
# 5. VISUALIZATION UTILITIES
# ------------------------------------------

def visualize_scan(results, violations, mu_vals, tau_vals, R_vals):
    """
    Produce a six-panel figure summarizing:
      1) I vs R at fixed τ
      2) I vs μ at fixed R
      3) QI bound vs μ
      4) I vs τ at fixed μ
      5) Count of violations vs μ
      6) Energy‐density profile at the best (μ,τ,R)
      
    Args:
        results (dict): Scan results
        violations (dict): Violation flags
        mu_vals (list): μ values scanned
        tau_vals (list): τ values scanned
        R_vals (list): R values scanned
        
    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15,10))
    plt.suptitle("3D Negative-Energy Shell Analysis", fontsize=16)

    # Panel 1: I vs R (μ var, τ fixed)
    ax1 = axes[0,0]
    tau0 = tau_vals[len(tau_vals)//2]
    for mu in mu_vals:
        I_R = [results[(mu,tau0,R)] for R in R_vals]
        ax1.plot(R_vals, I_R, 'o-', label=f'μ={mu:.2f}')
    ax1.set_xlabel("R")
    ax1.set_ylabel("I")
    ax1.set_title(f"I vs R (τ={tau0:.2f})")
    ax1.legend()
    ax1.grid(True)

    # Panel 2: I vs μ (τ var at fixed R)
    ax2 = axes[0,1]
    R0 = R_vals[len(R_vals)//2]
    for tau in tau_vals:
        I_μ = [results[(mu,tau,R0)] for mu in mu_vals]
        ax2.plot(mu_vals, I_μ, 's-', label=f'τ={tau:.2f}')
    ax2.set_xlabel("μ")
    ax2.set_ylabel("I")
    ax2.set_title(f"I vs μ (R={R0:.2f})")
    ax2.legend()
    ax2.grid(True)

    # Panel 3: QI bound vs μ
    ax3 = axes[0,2]
    bound_vals = [polymer_QI_bound(mu, tau0) for mu in mu_vals]
    ax3.plot(mu_vals, bound_vals, 'r-', label='QI bound')
    ax3.set_xlabel("μ")
    ax3.set_ylabel("Bound")
    ax3.set_title(f"QI Bound vs μ (τ={tau0:.2f})")
    ax3.legend()
    ax3.grid(True)

    # Panel 4: I vs τ (μ var at fixed R)
    ax4 = axes[1,0]
    mu0 = mu_vals[len(mu_vals)//2]
    for R in R_vals:
        I_τ = [results[(mu0,tau,R)] for tau in tau_vals]
        ax4.plot(tau_vals, I_τ, '^-', label=f'R={R:.2f}')
    ax4.set_xlabel("τ")
    ax4.set_ylabel("I")
    ax4.set_title(f"I vs τ (μ={mu0:.2f})")
    ax4.legend()
    ax4.grid(True)

    # Panel 5: Violation count vs μ
    ax5 = axes[1,1]
    counts = []
    for mu in mu_vals:
        c = sum(1 for (m,_,_) in violations if m==mu and violations[(m,_,_)] )
        counts.append(c)
    ax5.bar([f"{mu:.2f}" for mu in mu_vals], counts)
    ax5.set_xlabel("μ")
    ax5.set_ylabel("Count")
    ax5.set_title("Number of Violations per μ")
    ax5.grid(True, axis='y')

    # Panel 6: ρ(r) at optimal (μ,τ,R)
    ax6 = axes[1,2]
    best_key = min(results, key=lambda k: results[k])  # minimal I
    mu_best, tau_best, R_best = best_key
    sigma = 0.5
    A_best = 1.2*(np.pi/(2*mu_best))
    omega = 2*np.pi
    r_vals = np.linspace(0,8,200)
    pi_best = pi_shell(r_vals, R_best, sigma, A_best, omega, 0.0)
    ρ_best = energy_density_polymer(pi_best, mu_best)
    ax6.plot(r_vals, ρ_best, 'g-')
    ax6.axvline(R_best, color='r', linestyle='--', label=f'R={R_best:.2f}')
    ax6.set_xlabel("r")
    ax6.set_ylabel("ρ")
    ax6.set_title(f"ρ(r) at μ={mu_best:.2f}, τ={tau_best:.2f}, R={R_best:.2f}")
    ax6.legend()
    ax6.grid(True)

    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

    return fig

# ------------------------------------------
# 6. PLACEHOLDERS FOR 3+1D EVOLUTION & STABILITY
# ------------------------------------------

def evolve_phi_pi_3plus1D(phi_init, pi_init, grid_shape,
                          metric_params, mu, dt, dx, steps):
    """
    Placeholder: evolve (φ, π) on a 3D AMR grid with polymer corrections.
    Real implementation must solve:
      ∂φ/∂t = sin(μ π)/μ,
      ∂π/∂t = ∇² φ - m² φ + metric_coupling,
    on an adaptively refined mesh, coupled to Alcubierre metric solver.
    
    Args:
        phi_init (ndarray): Initial φ field
        pi_init (ndarray): Initial π field
        grid_shape (tuple): Grid dimensions
        metric_params (dict): Metric parameters
        mu (float): Polymer scale parameter
        dt (float): Time step
        dx (float): Spatial step
        steps (int): Number of evolution steps
        
    Returns:
        tuple: (phi_final, pi_final, evolution_data)
    """
    print("⚠️  evolve_phi_pi_3plus1D: Not yet implemented (requires full 3+1D solver).")
    return phi_init, pi_init, {}

def linearized_stability(phi_0, pi_0, mu, grid_shape, dt, dx, steps):
    """
    Placeholder: linearized stability analysis around (φ₀, π₀).
    Should compute eigenmodes of:
      δ̇φ = cos(μ π₀) δπ,
      δ̇π = ∇² δφ - m² δφ,
    and check for growing modes or superluminal signals.
    
    Args:
        phi_0 (ndarray): Background φ field
        pi_0 (ndarray): Background π field
        mu (float): Polymer scale parameter
        grid_shape (tuple): Grid dimensions
        dt (float): Time step
        dx (float): Spatial step
        steps (int): Number of steps for analysis
        
    Returns:
        dict: Stability analysis results
    """
    print("⚠️  linearized_stability: Not yet implemented (requires eigenvalue solver).")
    return {"stable": True, "max_growth_rate": 0.0, "unstable_modes": []}

def solve_warp_metric_3plus1D(r_grid, s_guess, phi, pi, mu, grid_shape):
    """
    Placeholder: solve Einstein equations with polymer T_{μν}^poly:
      R_{μν} - ½ g_{μν} R = 8π G T_{μν}^poly,
    for an Alcubierre ansatz:
      ds² = –[1 − v² s(r_b)] dt² − 2v s(r_b) dt dz + dx² + dy² + [1 + v² s(r_b)] dz².
      
    Args:
        r_grid (ndarray): Radial grid
        s_guess (function): Initial guess for shaping function
        phi (ndarray): φ field
        pi (ndarray): π field
        mu (float): Polymer scale parameter
        grid_shape (tuple): Grid dimensions
        
    Returns:
        tuple: (metric_solution, shaping_function, convergence_data)
    """
    print("⚠️  solve_warp_metric_3plus1D: Not yet implemented (requires GR solver).")
    return {}, s_guess, {}

# ------------------------------------------
# ANALYSIS UTILITIES
# ------------------------------------------

def generate_analysis_report(results, violations, mu_vals, tau_vals, R_vals,
                           squeezed_params=None):
    """
    Generate a comprehensive analysis report of the warp bubble scan results.
    
    Args:
        results (dict): Scan results
        violations (dict): Violation flags
        mu_vals (list): μ values scanned
        tau_vals (list): τ values scanned
        R_vals (list): R values scanned
        squeezed_params (dict, optional): Squeezed vacuum parameters
        
    Returns:
        str: Formatted analysis report
    """
    report = []
    report.append("=" * 60)
    report.append("WARP BUBBLE ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Summary statistics
    total_configs = len(results)
    violations_count = sum(violations.values())
    viable_count = total_configs - violations_count
    
    report.append(f"Total configurations tested: {total_configs}")
    report.append(f"Configurations violating QI bound: {violations_count}")
    report.append(f"Viable configurations: {viable_count}")
    report.append(f"Viability rate: {100*viable_count/total_configs:.1f}%")
    report.append("")
    
    # Best configurations
    viable_results = {k: v for k, v in results.items() if not violations[k]}
    if viable_results:
        best_config = min(viable_results, key=viable_results.get)
        mu_best, tau_best, R_best = best_config
        I_best = viable_results[best_config]
        bound_best = polymer_QI_bound(mu_best, tau_best)
        
        report.append("BEST VIABLE CONFIGURATION:")
        report.append(f"  μ = {mu_best:.3f}")
        report.append(f"  τ = {tau_best:.3f}")
        report.append(f"  R = {R_best:.3f}")
        report.append(f"  I = {I_best:.6e}")
        report.append(f"  QI Bound = {bound_best:.6e}")
        report.append(f"  Safety margin = {abs(bound_best/I_best):.2f}x")
    else:
        report.append("NO VIABLE CONFIGURATIONS FOUND")
        # Find least violating
        least_violating = min(results, key=lambda k: results[k] - polymer_QI_bound(k[0], k[1]))
        mu_lv, tau_lv, R_lv = least_violating
        I_lv = results[least_violating]
        bound_lv = polymer_QI_bound(mu_lv, tau_lv)
        
        report.append("")
        report.append("LEAST VIOLATING CONFIGURATION:")
        report.append(f"  μ = {mu_lv:.3f}")
        report.append(f"  τ = {tau_lv:.3f}")
        report.append(f"  R = {R_lv:.3f}")
        report.append(f"  I = {I_lv:.6e}")
        report.append(f"  QI Bound = {bound_lv:.6e}")
        report.append(f"  Violation ratio = {I_lv/bound_lv:.2f}")
    
    report.append("")
    
    # Parameter sensitivity analysis
    report.append("PARAMETER SENSITIVITY:")
      # μ analysis
    mu_violations = {}
    for mu in mu_vals:
        mu_violations[mu] = sum(1 for (m,t,r) in violations if m==mu and violations[(m,t,r)])
    best_mu = min(mu_violations, key=mu_violations.get)
    report.append(f"  Best μ (fewest violations): {best_mu:.3f} ({mu_violations[best_mu]} violations)")
    
    # τ analysis  
    tau_violations = {}
    for tau in tau_vals:
        tau_violations[tau] = sum(1 for (m,t,r) in violations if t==tau and violations[(m,t,r)])
    best_tau = min(tau_violations, key=tau_violations.get)
    report.append(f"  Best τ (fewest violations): {best_tau:.3f} ({tau_violations[best_tau]} violations)")
    
    # R analysis
    R_violations = {}
    for R in R_vals:
        R_violations[R] = sum(1 for (m,t,r) in violations if r==R and violations[(m,t,r)])
    best_R = min(R_violations, key=R_violations.get)
    report.append(f"  Best R (fewest violations): {best_R:.3f} ({R_violations[best_R]} violations)")
    
    if squeezed_params:
        report.append("")
        report.append("SQUEEZED VACUUM ANALYSIS:")
        E_req, E_squeeze = compare_neg_energy(**squeezed_params)
        feasibility = abs(E_squeeze/E_req) if E_req != 0 else float('inf')
        report.append(f"  Required energy: {E_req:.3e} J")
        report.append(f"  Available energy: {E_squeeze:.3e} J")
        report.append(f"  Feasibility ratio: {feasibility:.3e}")
        
        if feasibility >= 1.0:
            report.append("  STATUS: ENERGY REQUIREMENTS MET! 🚀")
        elif feasibility >= 0.1:
            report.append("  STATUS: Close to feasible (within order of magnitude)")
        else:
            report.append("  STATUS: Significant energy deficit")
    
    report.append("")
    report.append("=" * 60)
    
    return "\n".join(report)

# ------------------------------------------
# MAIN ROUTINE
# ------------------------------------------

def run_warp_analysis(mu_vals=None, tau_vals=None, R_vals=None, 
                     sigma=0.5, A_factor=1.2, omega=2*np.pi,
                     squeezed_analysis=True, generate_plots=True):
    """
    Run complete warp bubble analysis with default or custom parameters.
    
    Args:
        mu_vals (list, optional): μ values to scan
        tau_vals (list, optional): τ values to scan
        R_vals (list, optional): R values to scan
        sigma (float): Shell width parameter
        A_factor (float): Amplitude factor
        omega (float): Frequency
        squeezed_analysis (bool): Whether to perform squeezed vacuum analysis
        generate_plots (bool): Whether to generate visualization plots
        
    Returns:
        dict: Complete analysis results
    """
    # Default parameters
    if mu_vals is None:
        mu_vals = [0.1, 0.3, 0.6, 1.0]
    if tau_vals is None:
        tau_vals = [0.5, 1.0, 2.0]
    if R_vals is None:
        R_vals = [2.0, 3.0, 4.0]

    print("\n🔍 Running 3D Negative-Energy Shell Scan...")
    results, violations = scan_3d_shell(mu_vals, tau_vals, R_vals, sigma, A_factor, omega)

    # μ optimization
    print("\n🔧 Optimizing μ for QI bound...")
    best_mu, best_bound, mu_arr, bound_arr = find_optimal_mu()
    
    analysis_results = {
        'scan_results': results,
        'violations': violations,
        'parameters': {
            'mu_vals': mu_vals,
            'tau_vals': tau_vals, 
            'R_vals': R_vals,
            'sigma': sigma,
            'A_factor': A_factor,
            'omega': omega
        },
        'optimization': {
            'best_mu': best_mu,
            'best_bound': best_bound,
            'mu_array': mu_arr,
            'bound_array': bound_arr
        }
    }
    
    # Squeezed vacuum analysis
    if squeezed_analysis:
        print("\n⚡ Analyzing squeezed vacuum energy requirements...")
        squeezed_params = {
            'mu': best_mu,
            'tau': 1e-9,        # 1 ns sampling
            'R': 3.0,
            'dR': 0.5,
            'r_squeeze': 1.0,
            'omega': 2 * np.pi * 5e9,  # 5 GHz
            'cavity_vol': 1e-12  # 1 picoliter
        }
        
        E_req, E_squeeze = compare_neg_energy(**squeezed_params)
        analysis_results['squeezed_analysis'] = {
            'parameters': squeezed_params,
            'E_required': E_req,
            'E_available': E_squeeze,
            'feasibility_ratio': abs(E_squeeze/E_req) if E_req != 0 else float('inf')
        }
        
        print(f"Required E_neg (μ={best_mu:.3f}): {E_req:.3e} J")
        print(f"Squeezed vacuum ΔE:   {E_squeeze:.3e} J")
        print(f"Feasibility ratio: {abs(E_squeeze/E_req):.3e}")
    
    # Generate comprehensive report
    print("\n📊 Generating analysis report...")
    squeezed_params_for_report = analysis_results.get('squeezed_analysis', {}).get('parameters')
    report = generate_analysis_report(results, violations, mu_vals, tau_vals, R_vals,
                                    squeezed_params_for_report)
    analysis_results['report'] = report
    print(report)
    
    # Visualization
    if generate_plots:
        print("\n📈 Generating visualization...")
        fig = visualize_scan(results, violations, mu_vals, tau_vals, R_vals)
        analysis_results['figure'] = fig
    
    # Placeholder demonstrations
    print("\n🔮 Running placeholder demonstrations...")
    phi0 = np.zeros((50,50,50))
    pi0 = np.zeros((50,50,50))
    
    evolve_phi_pi_3plus1D(phi0, pi0, (50,50,50), {}, best_mu, dt=0.01, dx=0.1, steps=100)
    stability_result = linearized_stability(phi0, pi0, best_mu, (50,50,50), dt=0.01, dx=0.1, steps=100)
    metric_result = solve_warp_metric_3plus1D(np.linspace(0,5,50), lambda r: np.exp(-r**2), phi0, pi0, best_mu, (50,50,50))
    
    analysis_results['placeholders'] = {
        'stability': stability_result,
        'metric': metric_result
    }
    
    print("\n✅ Warp bubble analysis complete!")
    return analysis_results


if __name__ == "__main__":
    # Run the complete analysis
    results = run_warp_analysis()
