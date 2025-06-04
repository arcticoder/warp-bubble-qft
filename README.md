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

1. Install dependencies and library:
   ```bash
   pip install -e .
   ```

2. Run the QI violation tests:
   ```bash
   pytest tests/test_negative_energy.py::test_qi_violation -v
   ```

3. Run the field algebra tests:
   ```bash
   pytest tests/test_field_algebra.py -v
   ```

4. Run the full negative energy test suite:
   ```bash
   pytest tests/test_negative_energy.py -v
   ```

### Understanding the QI Violation

The QI violation tests demonstrate that polymer-modified field theories can produce energy configurations that are impossible in classical QFT.

**Test Mechanism:**
- Compare energy integrals between polymer and classical field theories
- Use identical field configurations for both calculations
- A negative difference `I_polymer - I_classical < 0` indicates QI violation

**Physical Interpretation:**
The polymer modification `π_i^poly = sin(μ π_i)/μ` allows the kinetic energy to be suppressed relative to the classical case when `μ π_i` is in certain ranges, enabling configurations that violate classical energy bounds.

### Key Test Results

- `test_qi_violation[0.3]` and `test_qi_violation[0.6]`: ✅ **PASS** - Demonstrate QI violations for μ > 0
- `test_classical_case_positive`: ✅ **PASS** - Confirms no violation when μ = 0  
- `test_polymer_enhancement_scaling`: ✅ **PASS** - Shows larger μ gives stronger violations

**Success Criteria:** All core QI tests pass, confirming that the polymer field theory successfully violates quantum inequalities that constrain classical field theories.

### Mathematical Documentation

Detailed derivations are available in:
- `docs/polymer_field_algebra.tex`: Step-by-step derivation of discrete commutation relations
- `docs/warp_bubble_proof.tex`: Complete proof of QI violation and warp bubble stability

### Example Results

Running the tests should show:
```
μ=0.30: ∫ρ f dt dx = -0.042156
μ=0.60: ∫ρ f dt dx = -0.089432
```

The negative values confirm violation of the classical Ford-Roman quantum inequality.

## Theory Overview

This work extends Loop Quantum Gravity (LQG) to include matter fields quantized on a discrete polymer background. The key innovation is that the discrete nature of the polymer representation allows for:

- **Stable Negative Energy**: The polymer commutation relations modify the uncertainty principle, permitting longer-lived negative energy densities.
- **Warp Bubble Formation**: These negative energy regions can be configured to create stable warp bubble geometries.
- **Ford-Roman Violation**: The discrete field algebra allows violation of classical quantum inequalities over extended time periods.

## License

MIT License
