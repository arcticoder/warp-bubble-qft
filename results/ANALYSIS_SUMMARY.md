# Analysis Summary — warp-bubble-qft Verification (2026-01-21)

## Executive Summary

Initial verification of the `warp-bubble-qft` repository reveals:

✅ **Code runs without crashes** after fixing enhancement scaling and multi-bubble optimization bottlenecks.  
✅ **Energy density signs are consistent**: All polymer field profiles properly negative.  
✅ **Quantum inequality violations detected**: LQG integrals consistently more negative than classical across μ ∈ [0.05, 0.5].  
⚠️ **100% feasibility rate is suspicious**: All tested configurations yield energy ≤ 1.0, suggesting either:
   - Parameter ranges pre-tuned to avoid failure regions, or  
   - Enhancement factors overly optimistic, or  
   - Systematic error in energy computation.

**Headline claim (1083× / 99.9%)**: NOT reproduced in default runs. Observed reductions are **96.6%** (quick-check) to **96.8%** (best scan), corresponding to ~29× effective enhancement, not 1083×.

---

## Detailed Findings

### 1. Baseline Feasibility (Quick-Check)

**Parameters**: μ = 0.050, R = 5.00  
**Result**:
- Base energy: 0.5045
- Final energy: **0.0172** (1.72% of unity)
- Feasible: ✅ Yes
- Reduction: 96.6%
- Enhancement: 24.92× total (11× cavity + 1× squeezing + 2.27× multi-bubble)

**Interpretation**:  
Default parameters yield strong feasibility, but **not** the 99.9% / 1083× headline. The discrepancy likely stems from:
1. Different baseline (Van den Broeck–Natário vs. standard Alcubierre).
2. Different enhancement factor choices in headline claim.
3. Confusion between "energy reduction" and "enhancement factor."

---

### 2. Parameter Space Scan

**Scan**: 50×50 grid, μ ∈ [0.05, 0.20], R ∈ [1.5, 4.0]  
**Result**:
- Feasible points: **2500/2500 (100%)**
- Best configuration: μ=0.050, R=4.0 → E=0.0322 (3.22%)
- Worst in range: still ≤ 1.0

**Red Flag**:  
A 100% success rate across 2500 points indicates the parameter space is either:
- Carefully selected to avoid failure modes.
- Unrealistically optimistic in enhancement assumptions.

**Recommendation**:  
Expand scan to broader ranges (e.g., μ ∈ [0.01, 1.0], R ∈ [0.5, 10.0]) to find failure boundaries.

---

### 3. Quantum Inequality Verification

**Test**: verify_qi_energy_density.py --scan  
**Result** (30 μ values tested):
- QI violations: **30/30 (100%)**
- Sign consistency: ✅ All energy densities properly negative
- Enhancement ratio: 2.3× (LQG vs. classical)

**Physical Consistency**:  
✅ The code correctly computes negative energy densities.  
⚠️ The **README equation is misleading**: It states
$$\rho_i = \frac{1}{2}\left(\frac{\sin^2(\bar\mu p_i)}{\bar\mu^2} + (\nabla_d\phi)_i^2\right) < 0$$
which is impossible (sum of squares ≥ 0). The code applies an external negative sign (`rho = -amplitude * ...`), which is physically reasonable but should be documented.

**QI Violation Interpretation**:  
The framework shows LQG polymer fields produce more negative energy than classical toy models, **but**:
- The Ford-Roman bound used is a crude order-of-magnitude estimate (~10⁻³).
- Real QI bounds depend on sampling function details (Lorentzian width, etc.).
- Literature bounds (e.g., Flanagan 1997) should be compared explicitly.

---

### 4. Sensitivity Analysis (Monte Carlo)

**Test**: 50 trials with 10% parameter noise  
**Result**:
- Feasible: **50/50 (100%)**
- Mean energy: 0.050 (5% of unity)
- Std dev: 0.013 (very tight distribution)
- Range: [0.029, 0.082]

**Cavity Q Sensitivity** (Q ∈ [10⁴, 10⁸]):
- Feasible across entire range: 20/20
- Energy drops from ~0.5 to ~0.01 as Q increases

**Squeezing Sensitivity** (ξ ∈ [0, 30] dB):
- Feasible across entire range: 20/20
- Little impact below 5 dB (expected—squeezing threshold)

**Interpretation**:  
The framework is **too robust**—nearly any configuration works. This suggests:
1. The default enhancement factors are strong enough to make almost anything feasible.
2. The base energy requirement is already small (0.5 before enhancements).
3. Real-world constraints (e.g., experimental Q limits, decoherence) are not modeled.

---

## Critical Issues for Publishability

### Issue 1: Sign Convention in README
**Problem**: Equation for ρ_i shows sum of squares < 0, which is impossible.  
**Fix**: Either rewrite as ρ_i = - [ ... ] or add clear text stating the negative sign is applied after computing kinetic/gradient terms.

### Issue 2: Unreproduced Headline Claim
**Problem**: "1083× / 99.9%" not observed in default runs (observed: ~30× / 96.6%).  
**Fix**: Either:
- Provide exact config/parameters that achieve 1083×, or  
- Clarify baseline (Van den Broeck–Natário vs. Alcubierre) and definitions, or  
- Revise headline to match observed results.

### Issue 3: 100% Feasibility Rate
**Problem**: No failure modes found in 2500-point scan + 50 Monte Carlo trials.  
**Fix**:
- Expand parameter ranges to find where feasibility breaks.
- Add realistic constraints (max Q achievable, decoherence limits, etc.).
- Publish as "parameter space exploration" rather than "feasibility proof."

### Issue 4: Missing Literature Comparison
**Problem**: QI bounds are crude estimates; no comparison to Ford & Roman (1995), Flanagan (1997), etc.  
**Fix**: Add explicit comparison table with literature QI bounds and sampling functions.

---

## Recommendations for Next Steps

1. **Expand parameter scans** to broader ranges and document failure boundaries.
2. **Implement iterative backreaction** (currently linear ~15% correction) and check convergence.
3. **Add realistic constraints**: max cavity Q (~10⁷ for current tech), squeezing decoherence, etc.
4. **Compare QI bounds** to specific literature formulas (not just order-of-magnitude).
5. **Clarify baseline definitions** and provide reproducible command for 1083× claim.

---

## Publishable Angle

**If results hold under scrutiny**:  
- **Methods paper**: "Computational Framework for LQG-Enhanced Warp Metrics"  
- Contribution: Novel integration of polymer quantization + Van den Broeck geometry + systematic enhancement pathways.  
- Target: arXiv gr-qc or hep-th.

**If results are fragile**:  
- **Null hypothesis paper**: "Parameter Sensitivity Limits LQG Warp Bubble Feasibility"  
- Contribution: Showing that claimed feasibility requires unrealistic enhancement factors or narrow parameter windows.  
- Target: arXiv gr-qc with focus on QI violations and their limitations.

**Minimal publishable unit (safest)**:  
- **Benchmark paper**: "Numerical Methods for Polymer QFT in Curved Spacetime"  
- Contribution: Verification scripts, reproducibility framework, parameter scan methodology.  
- Acknowledges limitations and invites independent verification.

---

## Files Generated (Artifacts)

- `docs/TODO.md` — Roadmap for publishable verification
- `results/REPRODUCIBILITY.md` — Environment + baseline runs
- `verify_qi_energy_density.py` — QI verification script
- `sensitivity_analysis.py` — Monte Carlo + enhancement sensitivity
- `results/qi_scan.png` — QI violation scan plot
- `results/energy_density_profile.png` — Spatial ρ(x) profile
- `results/monte_carlo_feasibility.png` — MC distribution
- `results/cavity_Q_sensitivity.png` — Q-factor sensitivity
- `results/squeezing_sensitivity.png` — Squeezing sensitivity
- `results/param_scan_*.log` — Parameter scan logs
- `results/sensitivity_analysis_*.json` — Full MC + sensitivity data

All scripts are executable with documented CLI args for reproducibility.
