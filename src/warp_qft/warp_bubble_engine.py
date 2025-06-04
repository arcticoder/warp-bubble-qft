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
    from scipy.integrate import simpson as simps
except ImportError:
    from scipy.integrate import simps
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
    """ρ_eff(r) = ½ [ (sin(μ π(r))/μ)² ]."""
    if mu == 0:
        return 0.5 * pi_r**2
    return 0.5 * (np.sin(mu * pi_r) / mu)**2

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
        radial_int = simps(rho_rt * vol_factor, r)
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
# MAIN ANALYSIS CLASS
# ------------------------------------------

class WarpBubbleEngine:
    """
    Comprehensive warp bubble analysis engine integrating all theoretical components.
    """
    
    def __init__(self):
        self.results = {}
        self.violations = {}
        self.optimal_params = {}
        
    def run_full_analysis(self, mu_vals: Optional[List[float]] = None,
                         tau_vals: Optional[List[float]] = None,
                         R_vals: Optional[List[float]] = None,
                         **kwargs) -> Dict:
        """
        Execute complete warp bubble feasibility analysis.
        """
        # Default parameters
        if mu_vals is None:
            mu_vals = [0.1, 0.3, 0.6, 1.0]
        if tau_vals is None:
            tau_vals = [0.5, 1.0, 2.0]
        if R_vals is None:
            R_vals = [2.0, 3.0, 4.0]
            
        sigma = kwargs.get('sigma', 0.5)
        A_factor = kwargs.get('A_factor', 1.2)
        omega = kwargs.get('omega', 2 * np.pi)
        
        print("🚀 Starting Comprehensive Warp Bubble Analysis")
        print("=" * 50)
        
        # 1. Shell scan
        print("\n🔍 Running 3D Negative-Energy Shell Scan...")
        self.results, self.violations = scan_3d_shell(
            mu_vals, tau_vals, R_vals, sigma, A_factor, omega
        )
        
        # 2. Optimization
        print("\n🔧 Optimizing polymer parameter μ...")
        best_mu, best_bound, mu_arr, bound_arr = find_optimal_mu()
        self.optimal_params = {
            'mu': best_mu,
            'bound': best_bound,
            'mu_array': mu_arr,
            'bound_array': bound_arr
        }
        
        # 3. Energy comparison
        print("\n⚡ Comparing required vs available negative energy...")
        mu_test = best_mu
        tau_test = 1e-9        # 1 ns sampling
        R_test = 3.0
        dR = 0.5
        r_squeeze = 1.0
        omega_mw = 2 * np.pi * 5e9  # 5 GHz
        cavity_vol = 1e-12  # 1 picoliter
        
        E_req, E_squeeze = compare_neg_energy(
            mu_test, tau_test, R_test, dR, r_squeeze, omega_mw, cavity_vol
        )
        
        # 4. Results summary
        analysis_results = {
            'scan_results': self.results,
            'violations': self.violations,
            'optimal_mu': best_mu,
            'optimal_bound': best_bound,
            'energy_required': E_req,
            'energy_available': E_squeeze,
            'feasibility_ratio': abs(E_squeeze / E_req) if E_req != 0 else np.inf,
            'parameters': {
                'mu_vals': mu_vals,
                'tau_vals': tau_vals,
                'R_vals': R_vals,
                'sigma': sigma,
                'A_factor': A_factor,
                'omega': omega
            }
        }
        
        self._print_summary(analysis_results)
        return analysis_results
    
    def _print_summary(self, results: Dict):
        """Print analysis summary."""
        print("\n📊 ANALYSIS SUMMARY")
        print("=" * 30)
        print(f"Optimal μ: {results['optimal_mu']:.3f}")
        print(f"Optimal bound: {results['optimal_bound']:.3e} J")
        print(f"Required energy: {results['energy_required']:.3e} J")
        print(f"Available energy: {results['energy_available']:.3e} J")
        print(f"Feasibility ratio: {results['feasibility_ratio']:.3e}")
        
        # Count violations
        total_violations = sum(1 for v in results['violations'].values() if v)
        total_configs = len(results['violations'])
        print(f"QI violations: {total_violations}/{total_configs} configurations")
        
        if results['feasibility_ratio'] > 1:
            print("\n✅ WARP BUBBLE POTENTIALLY FEASIBLE!")
        else:
            print(f"\n⚠️  More negative energy needed (factor: {1/results['feasibility_ratio']:.1f})")

# ------------------------------------------
# MAIN ROUTINE
# ------------------------------------------

def run_warp_bubble_analysis() -> Dict:
    """
    Complete warp bubble analysis pipeline.
    Returns comprehensive results dictionary.
    """
    # 1. Scanner parameters
    mu_vals  = [0.1, 0.3, 0.6, 1.0]
    tau_vals = [0.5, 1.0, 2.0]
    R_vals   = [2.0, 3.0, 4.0]
    sigma    = 0.5
    A_factor = 1.2
    omega    = 2 * np.pi

    print("\n🔍 Running 3D Negative-Energy Shell Scan...")
    results, violations = scan_3d_shell(mu_vals, tau_vals, R_vals, sigma, A_factor, omega)

    # Display a few key outcomes
    print("\n📊 Sample Results:")
    print("μ    τ    R    I(μ,τ,R)    Status")
    print("-" * 40)
    for (mu, tau, R) in [(0.3, 1.0, 2.0), (0.6, 1.0, 3.0)]:
        I_val = results[(mu, tau, R)]
        bound = polymer_QI_bound(mu, tau)
        status = "VIOLATES" if I_val < bound else "OK"
        print(f"{mu:.1f}  {tau:.1f}  {R:.1f}  {I_val:+.6e}  {status}")

    # 2. Optimize μ for the QI bound
    best_mu, best_bound, mu_arr, bound_arr = find_optimal_mu()
    print(f"\n🔧 Optimal μ ≈ {best_mu:.3f}, with bound = {best_bound:.3e} J")

    # 3. Compare required vs. available negative energy
    mu_test = best_mu
    tau_test = 1e-9        # 1 ns sampling
    R_test = 3.0
    dR = 0.5
    r_squeeze = 1.0
    omega_mw = 2 * np.pi * 5e9  # 5 GHz
    cavity_vol = 1e-12  # 1 picoliter

    E_req, E_squeeze = compare_neg_energy(mu_test, tau_test, R_test, dR, r_squeeze, omega_mw, cavity_vol)
    print(f"\nRequired E_neg (μ={mu_test:.3f}): {E_req:.3e} J")
    print(f"Squeezed vacuum ΔE:   {E_squeeze:.3e} J")
    print(f"Feasibility ratio (squeezed/required): {abs(E_squeeze/E_req):.3e}\n")

    # 4. Visualize the scan
    fig = visualize_scan(results, violations, mu_vals, tau_vals, R_vals)

    # 5. Placeholder calls
    phi0 = np.zeros((50,50,50)); pi0 = np.zeros((50,50,50))
    evolve_phi_pi_3plus1D(phi0, pi0, (50,50,50), {}, mu_test, dt=0.01, dx=0.1, steps=100)
    linearized_stability(phi0, pi0, mu_test, (50,50,50), dt=0.01, dx=0.1, steps=100)
    solve_warp_metric_3plus1D(np.linspace(0,5,50), lambda r: np.exp(-r**2), phi0, pi0, mu_test, (50,50,50))
if __name__ == "__main__":
    # Initialize and run analysis
    engine = WarpBubbleEngine()
    analysis = engine.run_full_analysis()
    
    # Generate visualization
    print("\n📈 Generating visualization...")
    fig = visualize_scan(
        analysis['scan_results'], 
        analysis['violations'],
        analysis['parameters']['mu_vals'],
        analysis['parameters']['tau_vals'], 
        analysis['parameters']['R_vals']
    )
    plt.show()
    
    # Placeholder demonstrations
    print("\n🔬 Running placeholder demonstrations...")
    phi0 = np.zeros((50, 50, 50))
    pi0 = np.zeros((50, 50, 50))
    
    evolve_phi_pi_3plus1D(phi0, pi0, (50, 50, 50), {}, 
                         analysis['optimal_mu'], dt=0.01, dx=0.1, steps=100)
    linearized_stability(phi0, pi0, analysis['optimal_mu'], 
                        (50, 50, 50), dt=0.01, dx=0.1, steps=100)
    solve_warp_metric_3plus1D(np.linspace(0, 5, 50), 
                             lambda r: np.exp(-r**2), 
                             phi0, pi0, analysis['optimal_mu'], (50, 50, 50))
    
    print("\n🎯 Analysis complete! Next steps:")
    print("   1. Implement full 3+1D PDE solver")
    print("   2. Add AMR grid capabilities") 
    print("   3. Couple to Einstein field equations")
    print("   4. Optimize experimental parameters")
