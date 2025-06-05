"""
Warp Bubble Engine: Comprehensive Implementation
==============================================

This module integrates:
1. Squeezed-vacuum negative energy estimation
2. 3D shell scan with Ford-Roman bound checks  
3. Polymer parameter optimization
4. Required vs available energy comparison
5. Placeholders for full 3+1D evolution and stability

Based on theoretical foundations in docs/*.tex
"""

import numpy as np
try:
    from scipy.integrate import simpson
except ImportError:
    try:
        from scipy.integrate import simps as simpson
    except ImportError:
        # Fallback implementation
        def simpson(y, x=None, dx=1.0, axis=-1):
            """Simple fallback for Simpson's rule integration."""
            return np.trapz(y, x=x, dx=dx, axis=axis)
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import warnings

# ------------------------------------------
# 1. SQUEEZED-VACUUM NEGATIVE-ENERGY ESTIMATE
# ------------------------------------------

def squeezed_vacuum_energy(r_squeeze: float, omega: float, volume: float, 
                          hbar: float = 1.055e-34) -> float:
    """
    Estimate the maximum negative energy density (J/m³) from a squeezed-vacuum state.
    Model: ρ_neg ≈ - (ħ * ω / volume) * sinh(r_squeeze).
    
    Args:
        r_squeeze: Squeezing parameter (dimensionless)
        omega: Angular frequency (rad/s)
        volume: Cavity volume (m³)
        hbar: Reduced Planck constant (J·s)
        
    Returns:
        Negative energy density in J/m³
    """
    return - (hbar * omega / volume) * np.sinh(r_squeeze)


# ------------------------------------------
# 2. 3D NEGATIVE-ENERGY SHELL SCAN
# ------------------------------------------

def sampling_function(t: np.ndarray, tau: float) -> np.ndarray:
    """Gaussian sampling: f(t) = exp(-t²/(2τ²)) / (sqrt(2π) τ)."""
    return np.exp(-t**2 / (2 * tau**2)) / (np.sqrt(2 * np.pi) * tau)

def pi_shell(r: np.ndarray, R: float, sigma: float, A: float, 
            omega: float, t: float) -> np.ndarray:
    """π(r,t) = A * exp(- (r - R)² / (2 σ²)) * sin(ω t)."""
    return A * np.exp(- ((r - R)**2) / (2 * sigma**2)) * np.sin(omega * t)

def energy_density_polymer(pi_r: np.ndarray, mu: float) -> np.ndarray:
    """ρ_eff(r) = ½ [ (sin(π μ π(r))/(π μ))² ] - CORRECTED SINC DEFINITION."""
    if mu == 0:
        return 0.5 * pi_r**2
    return 0.5 * (np.sin(np.pi * mu * pi_r) / (np.pi * mu))**2

def polymer_QI_bound(mu: float, tau: float = 1.0, 
                    hbar: float = 1.055e-34) -> float:
    """
    Polymer-modified Ford–Roman bound:
      Bound(μ,τ) = - (ħ * sin(μ)/μ) / (12 π τ²).
    """
    sinc_mu = 1.0 if mu == 0 else np.sin(mu)/mu
    return - (hbar * sinc_mu) / (12 * np.pi * tau**2)

def compute_I_3d(mu: float, tau: float, R: float, sigma: float, A: float, omega: float, 
                 r_max: float = 10.0, Nr: int = 300, t_max: float = 5.0, Nt: int = 400) -> float:
    """
    Compute I(μ,τ,R) = ∫_{r=0}^∞ ∫_{t=-T/2}^{T/2} ρ_eff(r,t) f(t) 4π r² dr dt.
    Returns a float approximation of I.
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
        radial_int = simpson(rho_rt * vol_factor, r)
        I_sum += radial_int * f_t * dt

    return I_sum

def scan_3d_shell(mu_vals: List[float], tau_vals: List[float], R_vals: List[float], 
                  sigma: float, A_factor: float, omega: float) -> Tuple[Dict, Dict]:
    """
    Scan μ ∈ mu_vals, τ ∈ tau_vals, R ∈ R_vals → compute I(μ,τ,R).
    Returns two dicts:
      results[(μ,τ,R)] = I_value
      violations[(μ,τ,R)] = True if I_value < polymer_QI_bound(μ,τ)
    """
    results    = {}
    violations = {}
    total = len(mu_vals)*len(tau_vals)*len(R_vals)
    count = 0

    for mu in mu_vals:
        A = A_factor * (np.pi/(2*mu)) if mu > 0 else A_factor  # ensure μπ > π/2
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

def find_optimal_mu(mu_min: float = 0.1, mu_max: float = 1.0, steps: int = 50, 
                   tau: float = 1.0) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Sample μ in [mu_min, mu_max] to find the most relaxed (most negative) QI bound.
    Returns (best_mu, best_bound, mu_array, bound_array).
    """
    mu_array = np.linspace(mu_min, mu_max, steps)
    bound_array = np.array([polymer_QI_bound(mu, tau) for mu in mu_array])
    idx = np.argmin(bound_array)
    return mu_array[idx], bound_array[idx], mu_array, bound_array

# ------------------------------------------
# 4. COMPARE REQUIRED VS. AVAILABLE NEGATIVE ENERGY
# ------------------------------------------

def required_negative_energy(mu: float, tau: float = 1.0, R: float = 3.0, 
                           dR: float = 0.5, hbar: float = 1.055e-34) -> float:
    """
    Rough estimate: E_req ≈ |Bound(μ,τ)| * (4π R² dR).
    """
    bound = polymer_QI_bound(mu, tau, hbar)
    shell_vol = 4 * np.pi * R**2 * dR
    return abs(bound) * shell_vol

def compare_neg_energy(mu: float, tau: float, R: float, dR: float, 
                      r_squeeze: float, omega: float, cavity_vol: float) -> Tuple[float, float]:
    """
    Compute (E_req, E_squeezed) for given parameters:
      E_req = required negative energy (J)
      E_squeezed = achievable by squeezed vacuum (J)
    """
    E_req = required_negative_energy(mu, tau, R, dR)
    ρ_sq = squeezed_vacuum_energy(r_squeeze, omega, cavity_vol)
    E_squeeze = ρ_sq * cavity_vol
    return E_req, E_squeeze

# ------------------------------------------
# 5. VISUALIZATION UTILITIES
# ------------------------------------------

def visualize_scan(results: Dict, violations: Dict, mu_vals: List[float], 
                  tau_vals: List[float], R_vals: List[float]) -> plt.Figure:
    """
    Produce a six-panel figure summarizing:
      1) I vs R at fixed τ
      2) I vs μ at fixed R
      3) QI bound vs μ
      4) I vs τ at fixed μ
      5) Count of violations vs μ
      6) Energy‐density profile at the best (μ,τ,R)
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
    A_best = 1.2*(np.pi/(2*mu_best)) if mu_best > 0 else 1.2
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

def evolve_phi_pi_3plus1D(phi_init: np.ndarray, pi_init: np.ndarray, grid_shape: Tuple[int, int, int],
                          metric_params: Dict, mu: float, dt: float, dx: float, steps: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Placeholder: evolve (φ, π) on a 3D AMR grid with polymer corrections.
    Real implementation must solve:
      ∂φ/∂t = sin(μ π)/μ,
      ∂π/∂t = ∇² φ - m² φ + metric_coupling,
    on an adaptively refined mesh, coupled to Alcubierre metric solver.
    """
    print("⚠️  evolve_phi_pi_3plus1D: Not yet implemented (requires full 3+1D solver).")
    return phi_init, pi_init, {}

def linearized_stability(phi_0: np.ndarray, pi_0: np.ndarray, mu: float, 
                        grid_shape: Tuple[int, int, int], dt: float, dx: float, steps: int) -> Dict:
    """
    Placeholder: linearized stability analysis around (φ₀, π₀).
    Should compute eigenmodes of:
      δ̇φ = cos(μ π₀) δπ,
      δ̇π = ∇² δφ - m² δφ,
    and check for growing modes or superluminal signals.
    """
    print("⚠️  linearized_stability: Not yet implemented (requires eigenvalue solver).")
    return {"stable": True, "max_growth_rate": 0.0, "unstable_modes": []}

def solve_warp_metric_3plus1D(r_grid: np.ndarray, s_guess: callable, phi: np.ndarray, 
                             pi: np.ndarray, mu: float, grid_shape: Tuple[int, int, int]) -> Tuple[Dict, callable, Dict]:
    """
    Placeholder: solve Einstein equations with polymer T_{μν}^poly:
      R_{μν} - ½ g_{μν} R = 8π G T_{μν}^poly,
    for an Alcubierre ansatz:
      ds² = –[1 − v² s(r_b)] dt² − 2v s(r_b) dt dz + dx² + dy² + [1 + v² s(r_b)] dz².
    """
    print("⚠️  solve_warp_metric_3plus1D: Not yet implemented (requires GR solver).")
    return {}, s_guess, {}

# ------------------------------------------
# 7. WARP BUBBLE POWER ANALYSIS
# ------------------------------------------

def toy_negative_energy_density(x: np.ndarray, mu: float, R: float, 
                               rho0: float = 1.0, sigma: Optional[float] = None) -> np.ndarray:
    """
    Toy model of a negative‐energy distribution inside radius R:
    ρ(x) = -ρ0 * exp[-(x/σ)²] * sinc(μ).
    
    Args:
        x: Spatial coordinates
        mu: Polymer scale parameter
        R: Bubble radius
        rho0: Peak negative energy density scale
        sigma: Spatial width (defaults to R/2)
        
    Returns:
        Negative energy density profile
    """
    if sigma is None:
        sigma = R / 2
    return -rho0 * np.exp(-(x**2)/(sigma**2)) * np.sinc(mu / np.pi)

def available_negative_energy(mu: float, tau: float, R: float, 
                            Nx: int = 200, Nt: int = 200) -> float:
    """
    Compute total negative energy by integrating ρ(x)*f(t) over x∈[-R,R] and t∈[-5τ,5τ].
    
    Args:
        mu: Polymer parameter
        tau: Sampling width
        R: Bubble radius
        Nx: Spatial grid points
        Nt: Temporal grid points
        
    Returns:
        Total available negative energy
    """
    x = np.linspace(-R, R, Nx)
    t = np.linspace(-5*tau, 5*tau, Nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # Precompute sampling function and spatial profile
    f_t = sampling_function(t, tau)                           # shape: (Nt,)
    rho_x = toy_negative_energy_density(x, mu, R)            # shape: (Nx,)

    # Total energy = ∫ (ρ(x) dx) * (∫ f(t) dt)
    total_rho = np.sum(rho_x) * dx            # ∫ ρ(x) dx
    total_f = np.sum(f_t) * dt                # ∫ f(t) dt (≈1 by normalization)
    return total_rho * total_f

def warp_energy_requirement(R: float, v: float = 1.0, c: float = 1.0) -> float:
    """
    Rough estimate of energy required to form a warp bubble of radius R at speed v:
    E_req ≈ α * R * v², with α ~ O(1) in Planck units.
    (This is a placeholder; replace with a more accurate integral over T00 for your metric.)
    
    Args:
        R: Bubble radius (in Planck units)
        v: Warp velocity (normalized to c)
        c: Speed of light (normalized to 1)
        
    Returns:
        Required energy for warp bubble formation
    """
    α = 1.0  # dimensionless prefactor—tweak based on detailed metric calculation
    return α * R * (v**2) / (c**2)

def compute_feasibility_ratio(mu: float, tau: float, R: float, v: float = 1.0,
                            Nx: int = 500, Nt: int = 500) -> Tuple[float, float, float]:
    """
    Compute the feasibility ratio E_avail/E_req for warp bubble formation.
    
    Args:
        mu: Polymer scale parameter
        tau: Sampling width
        R: Bubble radius
        v: Warp velocity
        Nx: Spatial grid resolution
        Nt: Temporal grid resolution
        
    Returns:
        Tuple of (E_avail, E_req, feasibility_ratio)
    """
    E_avail = available_negative_energy(mu, tau, R, Nx, Nt)
    E_req = warp_energy_requirement(R, v)
    
    if E_req == 0:
        feasibility_ratio = np.inf if E_avail < 0 else 0
    else:
        feasibility_ratio = abs(E_avail) / E_req
        
    return E_avail, E_req, feasibility_ratio

# ------------------------------------------
# ENHANCED POWER ANALYSIS FRAMEWORK
# ------------------------------------------

def parameter_scan_feasibility(mu_range: Tuple[float, float] = (0.1, 1.0),
                              R_range: Tuple[float, float] = (0.5, 5.0),
                              num_points: int = 20,
                              tau: float = 1.0,
                              v: float = 1.0) -> Dict:
    """
    Comprehensive parameter scan for warp bubble feasibility.
    
    Args:
        mu_range: Range of polymer parameters to scan
        R_range: Range of bubble radii to scan  
        num_points: Number of points per dimension
        tau: Sampling width
        v: Warp velocity
        
    Returns:
        Dictionary with scan results and optimal parameters
    """
    mu_vals = np.linspace(mu_range[0], mu_range[1], num_points)
    R_vals = np.linspace(R_range[0], R_range[1], num_points)
    
    # Initialize result arrays
    feasibility_grid = np.zeros((len(mu_vals), len(R_vals)))
    E_avail_grid = np.zeros((len(mu_vals), len(R_vals)))
    E_req_grid = np.zeros((len(mu_vals), len(R_vals)))
    
    best_ratio = 0
    best_params = None
    
    print(f"Scanning {num_points}×{num_points} parameter grid...")
    total_iterations = len(mu_vals) * len(R_vals)
    iteration = 0
    
    for i, mu in enumerate(mu_vals):
        for j, R in enumerate(R_vals):
            iteration += 1
            if iteration % 10 == 0:
                print(f"Progress: {iteration}/{total_iterations} ({100*iteration/total_iterations:.1f}%)")
                
            E_avail, E_req, ratio = compute_feasibility_ratio(mu, tau, R, v)
            
            feasibility_grid[i, j] = ratio
            E_avail_grid[i, j] = E_avail
            E_req_grid[i, j] = E_req
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_params = (mu, R)
    
    return {
        'mu_range': mu_range,
        'R_range': R_range,
        'mu_vals': mu_vals,
        'R_vals': R_vals,
        'feasibility_grid': feasibility_grid,
        'E_avail_grid': E_avail_grid,
        'E_req_grid': E_req_grid,
        'best_ratio': best_ratio,
        'best_params': best_params,
        'scan_parameters': {
            'num_points': num_points,
            'tau': tau,
            'v': v
        }
    }

def visualize_feasibility_scan(scan_results: Dict) -> plt.Figure:
    """
    Create comprehensive visualization of feasibility parameter scan.
    
    Args:
        scan_results: Results from parameter_scan_feasibility()
        
    Returns:
        Matplotlib figure with multiple analysis panels
    """
    fig = plt.figure(figsize=(16, 12))
    
    mu_vals = scan_results['mu_vals']
    R_vals = scan_results['R_vals']
    feasibility_grid = scan_results['feasibility_grid']
    E_avail_grid = scan_results['E_avail_grid']
    E_req_grid = scan_results['E_req_grid']
    params = scan_results['scan_parameters']
    
    # 1. Feasibility ratio heatmap
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(feasibility_grid, extent=[R_vals[0], R_vals[-1], mu_vals[0], mu_vals[-1]],
                     aspect='auto', origin='lower', cmap='RdYlGn', vmin=0, vmax=2)
    ax1.set_xlabel('Bubble Radius R (Planck lengths)')
    ax1.set_ylabel('Polymer Parameter μ')
    ax1.set_title('Feasibility Ratio: E_avail/E_req')
    plt.colorbar(im1, ax=ax1)
    
    # Add feasibility threshold line
    ax1.contour(R_vals, mu_vals, feasibility_grid, levels=[1.0], colors='red', linewidths=2)
    
    # Mark best point
    if scan_results['best_params']:
        mu_best, R_best = scan_results['best_params']
        ax1.plot(R_best, mu_best, 'r*', markersize=15, markeredgecolor='black', 
                label=f'Best: μ={mu_best:.3f}, R={R_best:.3f}')
        ax1.legend()
    
    # 2. Available energy heatmap
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(np.abs(E_avail_grid), extent=[R_vals[0], R_vals[-1], mu_vals[0], mu_vals[-1]],
                     aspect='auto', origin='lower', cmap='Blues')
    ax2.set_xlabel('Bubble Radius R')
    ax2.set_ylabel('Polymer Parameter μ')
    ax2.set_title('Available Negative Energy |E_avail|')
    plt.colorbar(im2, ax=ax2, format='%.2e')
    
    # 3. Required energy heatmap
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(E_req_grid, extent=[R_vals[0], R_vals[-1], mu_vals[0], mu_vals[-1]],
                     aspect='auto', origin='lower', cmap='Reds')
    ax3.set_xlabel('Bubble Radius R')
    ax3.set_ylabel('Polymer Parameter μ')
    ax3.set_title('Required Energy E_req')
    plt.colorbar(im3, ax=ax3, format='%.2e')
    
    # 4. Feasibility vs mu (at optimal R)
    ax4 = plt.subplot(2, 3, 4)
    if scan_results['best_params']:
        _, R_opt = scan_results['best_params']
        R_idx = np.argmin(np.abs(R_vals - R_opt))
        ax4.plot(mu_vals, feasibility_grid[:, R_idx], 'b-', linewidth=2, 
                label=f'R = {R_opt:.3f}')
    else:
        # Use middle R value
        R_idx = len(R_vals) // 2
        ax4.plot(mu_vals, feasibility_grid[:, R_idx], 'b-', linewidth=2,
                label=f'R = {R_vals[R_idx]:.3f}')
    
    ax4.axhline(y=1.0, color='red', linestyle='--', label='Feasibility Threshold')
    ax4.set_xlabel('Polymer Parameter μ')
    ax4.set_ylabel('Feasibility Ratio')
    ax4.set_title('Feasibility vs Polymer Parameter')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Feasibility vs R (at optimal mu)
    ax5 = plt.subplot(2, 3, 5)
    if scan_results['best_params']:
        mu_opt, _ = scan_results['best_params']
        mu_idx = np.argmin(np.abs(mu_vals - mu_opt))
        ax5.plot(R_vals, feasibility_grid[mu_idx, :], 'g-', linewidth=2,
                label=f'μ = {mu_opt:.3f}')
    else:
        # Use middle mu value
        mu_idx = len(mu_vals) // 2
        ax5.plot(R_vals, feasibility_grid[mu_idx, :], 'g-', linewidth=2,
                label=f'μ = {mu_vals[mu_idx]:.3f}')
        
    ax5.axhline(y=1.0, color='red', linestyle='--', label='Feasibility Threshold')
    ax5.set_xlabel('Bubble Radius R')
    ax5.set_ylabel('Feasibility Ratio')
    ax5.set_title('Feasibility vs Bubble Radius')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate summary statistics
    max_ratio = np.max(feasibility_grid)
    feasible_fraction = np.sum(feasibility_grid >= 1.0) / feasibility_grid.size
    median_ratio = np.median(feasibility_grid)
    
    summary_text = f"""Parameter Scan Summary
    
Grid Resolution: {len(mu_vals)}×{len(R_vals)}
Parameter Ranges:
  μ: [{mu_vals[0]:.2f}, {mu_vals[-1]:.2f}]
  R: [{R_vals[0]:.2f}, {R_vals[-1]:.2f}]
  τ: {params['tau']:.2f}
  v: {params['v']:.2f}

Feasibility Statistics:
  Maximum Ratio: {max_ratio:.3f}
  Median Ratio: {median_ratio:.3f}
  Feasible Fraction: {feasible_fraction*100:.1f}%

Best Configuration:"""
    
    if scan_results['best_params']:
        mu_best, R_best = scan_results['best_params']
        summary_text += f"""
  μ_best: {mu_best:.3f}
  R_best: {R_best:.3f}
  Ratio: {scan_results['best_ratio']:.3f}
  
Status: {'✅ FEASIBLE' if scan_results['best_ratio'] >= 1.0 else '⚠️ INSUFFICIENT'}"""
    else:
        summary_text += "\n  No viable parameters found"
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.suptitle('Warp Bubble Feasibility Analysis', y=0.98, fontsize=16, fontweight='bold')
    
    return fig

def print_feasibility_summary(scan_results: Dict):
    """Print a comprehensive summary of feasibility scan results."""
    print("\n" + "="*60)
    print("🔬 WARP BUBBLE FEASIBILITY ANALYSIS SUMMARY")
    print("="*60)
    
    params = scan_results['scan_parameters']
    print(f"📊 Scan Parameters:")
    print(f"   Grid size: {len(scan_results['mu_vals'])}×{len(scan_results['R_vals'])}")
    print(f"   μ range: [{scan_results['mu_range'][0]:.2f}, {scan_results['mu_range'][1]:.2f}]")
    print(f"   R range: [{scan_results['R_range'][0]:.2f}, {scan_results['R_range'][1]:.2f}]")
    print(f"   τ = {params['tau']:.2f}, v = {params['v']:.2f}")
    
    # Statistics
    feasibility_grid = scan_results['feasibility_grid']
    max_ratio = np.max(feasibility_grid)
    feasible_count = np.sum(feasibility_grid >= 1.0)
    total_count = feasibility_grid.size
    
    print(f"\n📈 Results:")
    print(f"   Maximum feasibility ratio: {max_ratio:.3f}")
    print(f"   Feasible configurations: {feasible_count}/{total_count} ({100*feasible_count/total_count:.1f}%)")
    print(f"   Median feasibility ratio: {np.median(feasibility_grid):.3f}")
    
    if scan_results['best_params']:
        mu_best, R_best = scan_results['best_params']
        print(f"\n🎯 OPTIMAL CONFIGURATION:")
        print(f"   μ_optimal = {mu_best:.3f}")
        print(f"   R_optimal = {R_best:.3f} Planck lengths")
        print(f"   Feasibility ratio = {scan_results['best_ratio']:.3f}")
        
        if scan_results['best_ratio'] >= 1.0:
            surplus_factor = scan_results['best_ratio']
            print(f"\n✅ WARP BUBBLE APPEARS FEASIBLE!")
            print(f"   Energy surplus: {surplus_factor:.2f}x required")
            print("   🚀 Ready for next implementation phase!")
        else:
            shortage_factor = 1.0 / scan_results['best_ratio']
            print(f"\n⚠️  ADDITIONAL NEGATIVE ENERGY NEEDED")
            print(f"   Shortage factor: {shortage_factor:.1f}x")
            print("   Consider: higher μ, optimized sampling, or cavity enhancement.")
    
    print("\n🔬 Next experimental steps:")
    if scan_results['best_params']:
        mu_best, R_best = scan_results['best_params']
        print(f"   1. Target polymer parameter μ ≈ {mu_best:.3f}")
        print(f"   2. Design bubble with radius R ≈ {R_best:.3f} Planck lengths")
        print(f"   3. Optimize sampling width τ < {params['tau']:.3f}")
        print("   4. Implement cavity-enhanced negative energy generation")
    
    print("="*60)

def generate_energy_profile_analysis(mu: float, tau: float, R: float) -> Dict:
    """
    Generate detailed energy profile analysis for a specific configuration.
    
    Args:
        mu: Polymer parameter
        tau: Sampling width  
        R: Bubble radius
        
    Returns:
        Dictionary with detailed energy profile data
    """
    # High-resolution grids for analysis
    x = np.linspace(-R, R, 500)
    t = np.linspace(-5*tau, 5*tau, 500)
    
    # Compute energy density profile
    rho_x = toy_negative_energy_density(x, mu, R)
    
    # Temporal sampling function
    f_t = sampling_function(t, tau)
    
    # Find key characteristics
    peak_density_idx = np.argmin(rho_x)
    peak_density = rho_x[peak_density_idx]
    peak_position = x[peak_density_idx]
    
    # Bubble width (FWHM of negative region)
    negative_mask = rho_x < peak_density/2
    if np.any(negative_mask):
        negative_indices = np.where(negative_mask)[0]
        bubble_width = x[negative_indices[-1]] - x[negative_indices[0]]
    else:
        bubble_width = 0.0
    
    # Energy integral components
    spatial_integral = np.trapz(rho_x, x)
    temporal_integral = np.trapz(f_t, t)
    total_energy = spatial_integral * temporal_integral
    
    # Quantum inequality bound
    qi_bound = polymer_QI_bound(mu, tau)
    
    return {
        'spatial_grid': x,
        'temporal_grid': t,
        'energy_density': rho_x,
        'sampling_function': f_t,
        'peak_density': peak_density,
        'peak_position': peak_position,
        'bubble_width': bubble_width,
        'spatial_integral': spatial_integral,
        'temporal_integral': temporal_integral,
        'total_energy': total_energy,
        'qi_bound': qi_bound,
        'violates_qi': total_energy < qi_bound,
        'parameters': {'mu': mu, 'tau': tau, 'R': R}
    }

# ------------------------------------------
# COMPREHENSIVE POWER ANALYSIS METHODS
# ------------------------------------------

def run_power_analysis(mu_range: Tuple[float, float] = (0.1, 1.0),
                          R_range: Tuple[float, float] = (0.5, 5.0),
                          num_points: int = 20,
                          tau: float = 1.0,
                          v: float = 1.0,
                          visualize: bool = True) -> Dict:
        """
        Run comprehensive warp bubble power analysis.
        
        This method implements the core functionality requested by the user:
        1. Parameter space scanning for optimal configurations
        2. Available vs required energy comparison 
        3. Feasibility ratio calculation across parameter space
        4. Visualization of results
        
        Args:
            mu_range: Range of polymer parameters (μ) to scan
            R_range: Range of bubble radii to scan
            num_points: Resolution of parameter grid
            tau: Sampling width for temporal integration
            v: Warp velocity (normalized to c)
            visualize: Whether to generate plots
            
        Returns:
            Comprehensive analysis results dictionary
        """
        print("🚀 WARP BUBBLE POWER ANALYSIS")
        print("Quantifying negative energy requirements vs availability")
        print("="*60)
        
        print(f"\n🔍 Analysis Parameters:")
        print(f"   μ range: [{mu_range[0]:.2f}, {mu_range[1]:.2f}]")
        print(f"   R range: [{R_range[0]:.2f}, {R_range[1]:.2f}] Planck lengths")
        print(f"   Grid resolution: {num_points}×{num_points}")
        print(f"   Sampling width τ: {tau:.2f}")
        print(f"   Warp velocity v: {v:.2f}c")
        
        # Run parameter scan
        print("\n🔍 Scanning parameter space for feasibility...")
        self.feasibility_results = parameter_scan_feasibility(
            mu_range, R_range, num_points, tau, v
        )
        
        # Print summary
        print_feasibility_summary(self.feasibility_results)
        
        # Generate visualization
        if visualize:
            print("\n📈 Generating feasibility visualization...")
            fig = visualize_feasibility_scan(self.feasibility_results)
            plt.show()
        
        return self.feasibility_results

def analyze_specific_configuration(mu: float, tau: float, R: float, 
                                     v: float = 1.0, verbose: bool = True) -> Dict:
        """
        Analyze a specific warp bubble configuration in detail.
        
        Args:
            mu: Polymer parameter
            tau: Sampling width
            R: Bubble radius
            v: Warp velocity
            verbose: Whether to print detailed results
            
        Returns:
            Configuration analysis results
        """
        if verbose:
            print(f"\n🔬 Analyzing configuration: μ={mu:.3f}, τ={tau:.3f}, R={R:.3f}, v={v:.3f}")
        
        # Compute energies
        E_avail, E_req, feasibility_ratio = compute_feasibility_ratio(mu, tau, R, v)
        
        # Compute QI bound
        qi_bound = polymer_QI_bound(mu, tau)
        
        # Generate energy profile
        profile_analysis = generate_energy_profile_analysis(mu, tau, R)
        
        # Generate spatial profile for visualization
        x = np.linspace(-R, R, 200)
        energy_profile = toy_negative_energy_density(x, mu, R)
        
        results = {
            'parameters': {'mu': mu, 'tau': tau, 'R': R, 'v': v},
            'available_energy': E_avail,
            'required_energy': E_req,
            'feasibility_ratio': feasibility_ratio,
            'qi_bound': qi_bound,
            'energy_profile': energy_profile,
            'spatial_grid': x,
            'profile_analysis': profile_analysis,
            'is_feasible': feasibility_ratio >= 1.0,
            'violates_qi': E_avail < qi_bound
        }
        
        if verbose:
            print(f"  Available energy: {E_avail:.3e}")
            print(f"  Required energy:  {E_req:.3e}")
            print(f"  Feasibility ratio: {feasibility_ratio:.3f}")
            print(f"  QI bound: {qi_bound:.3e}")
            print(f"  Feasible: {'✅ YES' if results['is_feasible'] else '❌ NO'}")
            print(f"  QI violation: {'⚠️ YES' if results['violates_qi'] else '✅ NO'}")
        
        return results

def optimize_for_feasibility(target_ratio: float = 1.0,
                                mu_range: Tuple[float, float] = (0.1, 1.0),
                                R_range: Tuple[float, float] = (0.5, 5.0),
                                max_iterations: int = 50) -> Dict:
        """
        Optimize parameters to achieve target feasibility ratio.
        
        Args:
            target_ratio: Target feasibility ratio (1.0 = barely feasible)
            mu_range: Search range for mu parameter
            R_range: Search range for R parameter  
            max_iterations: Maximum optimization iterations
            
        Returns:
            Optimization results with best parameters found
        """
        print(f"\n🎯 Optimizing for feasibility ratio ≥ {target_ratio:.2f}")
        
        best_ratio = 0
        best_params = None
        iteration_data = []
        
        # Simple grid refinement optimization
        for iteration in range(max_iterations):
            # Sample parameters
            mu = np.random.uniform(mu_range[0], mu_range[1])
            R = np.random.uniform(R_range[0], R_range[1])
            tau = 1.0  # Fixed for now
            v = 1.0    # Fixed for now
            
            # Evaluate configuration
            E_avail, E_req, ratio = compute_feasibility_ratio(mu, tau, R, v)
            
            iteration_data.append({
                'iteration': iteration,
                'mu': mu, 'R': R, 'tau': tau, 'v': v,
                'ratio': ratio, 'E_avail': E_avail, 'E_req': E_req
            })
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_params = (mu, R)
                
                if ratio >= target_ratio:
                    print(f"✅ Target achieved at iteration {iteration}!")
                    break
