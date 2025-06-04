# Warp Bubble QFT

This repository contains the implementation of a quantum field theory on a polymer/loop background, focusing on generating stable negative-energy densities (warp bubbles) that can remain in violation of the Ford–Roman quantum inequalities.

## Essential Precursor Research Milestones from Unified LQG
% 1. AMR Error Estimator & Refinement
\[
  \eta_{\rm grad}(p) 
  = \sqrt{\Bigl(\frac{\partial \Phi}{\partial x}\Bigr)^2 
  + \Bigl(\frac{\partial \Phi}{\partial y}\Bigr)^2}\Big|_p,
  \quad
  \eta_{\rm curv}(p) 
  = \sqrt{\Bigl(\frac{\partial^2 \Phi}{\partial x^2} 
  + \frac{\partial^2 \Phi}{\partial y^2}\Bigr)^2}\Big|_p.
\]

% 2. Holonomy Substitutions
\[
  K_x \mapsto \frac{\sin(\bar\mu\,K_x)}{\bar\mu},
  \qquad
  K_\varphi \mapsto \frac{\sin(\bar\mu\,K_\varphi)}{\bar\mu}.
\]

% 3. Universal Polymer Resummation
\[
  f_{LQG}(r) 
  = 1 - \frac{2M}{r} 
  + \frac{\mu^{2}M^{2}}{6\,r^{4}}
    \frac{1}{\,1 + \frac{\mu^{4}M^{2}}{420\,r^{6}}\,} 
  + \mathcal{O}(\mu^{8}).
\]

% 4. Constraint Entanglement Measure
\[
  E_{AB} 
  = \langle\Psi\,|\,\hat H[N_A]\,\hat H[N_B]\,|\Psi\rangle
    \;-\;\langle\Psi\,|\,\hat H[N_A]\,|\Psi\rangle\,
    \langle\Psi\,|\,\hat H[N_B]\,|\Psi\rangle.
\]

% 5. Matter–Spacetime Duality Map
\[
  \delta E^x_i = \alpha\,\phi_i,\quad 
  \delta K_x^i = \frac{1}{\alpha}\,\pi_i,
  \quad
  \alpha = \sqrt{\frac{\hbar}{\gamma}}.
\]

% 6. Quantum Geometry Catalysis Factor
\[
  v_{\rm eff} = \Xi\,v_{\rm classical}, 
  \quad
  \Xi = 1 + \beta\,\frac{\ell_{\rm Pl}}{L_{\rm packet}},
  \quad 
  \ell_{\rm Pl} = 10^{-3},\;L_{\rm packet}=0.1,\;\beta\approx0.5.
\]

% 7. Negative-Energy Density (Warp Bubble)
\[
  \rho_i = \frac{1}{2}\Bigl(
    \frac{\sin^2(\bar\mu\,p_i)}{\bar\mu^2} + (\nabla_d \phi)_i^2
  \Bigr) \;<\; 0
  \quad\text{over } \Delta t \text{ beyond Ford–Roman bound.}
\]

% 8. Constraint Algebra (1D Midisuperspace)
\[
  [\,\hat H(N),\,\hat H(M)\,] 
  = i\hbar\,\hat D\bigl[q^{rr}(N\,M' - M\,N')\bigr],
  \quad
  [\hat H(N),\,\hat D(S^r)] 
  = i\hbar\,\hat H\bigl[S^r\,N'\bigr],
  \quad
  [\hat D(S^r),\,\hat D(T^r)] 
  = i\hbar\,\hat D\bigl[S^r\,T'^r - T^r\,S'^r\bigr].
\]


## Repository Structure

- `src/warp_qft/`  
  Core Python modules for polymerized field algebra and negative-energy stability.

- `tests/`  
  Unit tests for the discrete field commutators and negative-energy calculations.

- `docs/`  
  LaTeX derivations and documentation (e.g., `polymer_field_algebra.tex` and `warp_bubble_proof.tex`).

- `examples/`  
  Demonstration scripts and Jupyter notebooks (e.g., `demo_negative_energy.ipynb`, `demo_warp_bubble_sim.py`).

## Getting Started

1. Clone this repository:
   ```bash
   git clone <your-url>/warp-bubble-qft.git
   cd warp-bubble-qft
   ```

2. Install dependencies (Python 3.8+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

3. Run tests:
   ```bash
   pytest
   ```

4. Take a look at `docs/polymer_field_algebra.tex` for the derivations of discrete field commutators.

## Testing Quantum Inequality Violation

### Installation and Setup

1. Install dependencies:
   ```bash
   pip install -e .
   ```

2. Install additional dependencies for symbolic computation:
   ```bash
   pip install sympy matplotlib
   ```

### Running Tests

1. Run field-algebra tests:
   ```bash
   pytest tests/test_field_algebra.py -v
   ```

2. Run field commutator tests:
   ```bash
   pytest tests/test_field_commutators.py -v
   ```

3. Run QI violation test:
   ```bash
   pytest tests/test_negative_energy.py::test_qi_violation -v
   ```

4. Run full negative-energy suite:
   ```bash
   pytest tests/test_negative_energy.py -v
   ```

5. Run stability analysis tests:
   ```bash
   pytest tests/test_negative_energy_bounds.py -v
   ```

6. Run all tests:
   ```bash
   pytest -v
   ```

### Symbolic Derivation

Run the symbolic derivation of polymer-modified Ford-Roman bounds:
```bash
python scripts/qi_bound_symbolic.py
```

### Example Results

Running the quantum inequality violation tests should produce output similar to:

```text
μ = 0.30: ∫ρ f dt dx = –0.042156
μ = 0.60: ∫ρ f dt dx = –0.089432
μ = 1.00: ∫ρ f dt dx = –0.210987
```

These negative values demonstrate quantum inequality violations that are:
- **Forbidden** in classical field theory (μ = 0)
- **Permitted** in polymer field theory (μ > 0)

### Key Documentation

The theoretical foundations are documented in:
- `docs/qi_discrete_commutation.tex` - Rigorous small-μ expansion showing sinc-factor cancellation
- `docs/qi_bound_modification.tex` - Derivation of polymer-modified Ford-Roman bound  
- `docs/qi_numerical_results.tex` - Numerical demonstration of QI violations

## Theory Overview

This work extends Loop Quantum Gravity (LQG) to include matter fields quantized on a discrete polymer background. The key innovation is that the discrete nature of the polymer representation allows for:

- **Stable Negative Energy**: The polymer commutation relations modify the uncertainty principle, permitting longer-lived negative energy densities.
- **Warp Bubble Formation**: These negative energy regions can be configured to create stable warp bubble geometries.
- **Ford-Roman Violation**: The discrete field algebra allows violation of classical quantum inequalities over extended time periods.

## License

MIT License
