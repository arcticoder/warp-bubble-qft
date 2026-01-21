# TODO — Publishable-Grade Verification & Extensions (warp-bubble-qft)

This file is a working roadmap to turn the current research-stage framework into a reproducible, publishable-quality result (positive, negative/null, or methods/benchmark paper).

Guiding principle: **treat all headline claims as hypotheses** until reproduced under controlled sweeps with archived configs, versions, and deterministic outputs.

---

## 0) Reproducibility Baseline (do first)

- [ ] Record runtime environment
  - [ ] Capture `python --version`, `pip freeze`, OS info
  - [ ] Record commit hash and dirty state (`git status`)
- [ ] Establish a single “golden run” command + frozen config
  - Target script: `run_enhanced_lqg_pipeline.py`
  - Prefer writing results to a timestamped JSON file
- [ ] Define output artifacts to archive
  - [ ] `*.json` outputs
  - [ ] key plots (`*.png`)
  - [ ] logs (e.g. `lqg_pipeline.log`)

Deliverable: a short reproducibility note (what to run, what you get, hashes).

---

## 1) Reproduce & Verify Core Claims

### 1.1 Reproduce the “1083× / 99.9%” energy optimization headline

- [ ] Run quick feasibility check
  - Command: `python run_enhanced_lqg_pipeline.py --quick-check`
  - Record: base energy, final energy, enhancement breakdown
- [ ] Run full pipeline once (no scanning) with explicit output file
  - Command: `python run_enhanced_lqg_pipeline.py --complete --output results/repro_run.json`
  - If `results/` doesn’t exist, create it.
- [ ] Confirm how the repo defines “energy” and “ratio”
  - Identify the baseline (e.g., 3.80 GJ) and units consistency
  - Identify where the 1083× factor is computed

Publishable angle:
- **Verification paper** if the results reproduce.
- **Null hypothesis / parameter fragility** if they do not reproduce across small perturbations.

### 1.2 Verify QI-violation computation (Ford–Roman-style checks)

- [ ] Locate the canonical QI test implementation (likely under `tests/` or `src/warp_qft/`)
- [ ] Re-run QI checks with fixed random seeds (if any)
- [ ] Validate sign conventions and physical consistency
  - Note: the README includes an expression of the form
    $$\rho_i = \tfrac{1}{2}\left(\tfrac{\sin^2(\bar\mu p_i)}{\bar\mu^2} + (\nabla_d\phi)_i^2\right) < 0$$
    which is **non-negative as written** (sum of squares). Treat this as a red-flag item to reconcile with the code’s sign conventions.

Deliverable: a minimal, standalone script that recomputes the QI integral(s) from saved run outputs.

---

## 2) Sensitivity Analysis & Parameter Robustness

Goal: determine whether feasibility/optimization is robust across realistic ranges and noise.

- [ ] Reproduce parameter scans using built-in tools
  - Candidate scripts in repo root:
    - `fast_parameter_scan.py`
    - `optimized_parameter_scan.py`
    - `ultra_fast_scan.py`
- [ ] Add a Monte Carlo sweep (μ, R) with noise on initial conditions
  - Suggested ranges: μ ∈ [0.05, 0.5], R ∈ [1.0, 5.0]
  - Use seeded RNG for reproducibility
- [ ] Report robustness statistics
  - % feasible configurations
  - best/median/worst energy ratio
  - sensitivity of optimum to μ and R

Publishable angle:
- “Robustness (or fragility) of LQG-enhanced warp-bubble optimizations.”

---

## 3) Backreaction: Linear vs Nonlinear / Iterative Coupling

The repo mentions linear backreaction (~15% reduction). Upgrade this into a testable, convergent iterative loop.

- [ ] Locate backreaction implementation (`src/warp_qft/backreaction_solver.py`)
- [ ] Confirm whether coupling is applied as a single multiplicative factor or iterated
- [ ] Implement/verify an iterative update scheme with convergence criteria
  - stop when $|E_{n+1}-E_n|/|E_n| < \epsilon$
  - record iterations-to-converge and failures

Publishable angle:
- **Null result** if nonlinear/iterative coupling removes feasibility.
- **Methods** if a stable convergent solver is demonstrated.

---

## 4) Toward 3+1D (Scoped and Honest)

A full 3+1D evolution is likely out of scope short-term; do the smallest honest step that adds value.

- [ ] Add a toy 1D or 2D evolution harness (explicitly labeled as toy)
- [ ] Add stability diagnostics
  - divergence detection
  - CFL-like timestep checks (if using finite differences)
- [ ] Document what is *not* modeled (no full GR gauge conditions, etc.)

Publishable angle:
- “Failure modes / instability signatures under time evolution.”

---

## 5) Quantum-Optics Analogies (Squeezing) & Causality Checks

- [ ] Verify how squeezing is modeled in `src/warp_qft/enhancement_pathway.py`
- [ ] Decide whether to keep it as an effective factor or implement a minimal explicit model
- [ ] Add causality/CTC screening hooks (even if coarse)
  - clearly state assumptions and limitations

Publishable angle:
- “Causality constraints in polymer-enhanced warp metrics” (likely null/constraints-heavy).

---

## 6) Benchmark Against Literature / Known Bounds

- [ ] Add a comparison table in docs
  - what bounds are being used
  - what integral/averaging procedure is applied
  - parameter mapping between code and literature
- [ ] Ensure the paper-style narrative acknowledges known objections and limitations

Deliverable: a small “Methods & limitations” doc section suitable for arXiv.

---

## Working Notes (keep updated)

- Current recommended entrypoint: `run_enhanced_lqg_pipeline.py`
- Existing artifacts in repo root worth reviewing:
  - `enhanced_pipeline_results.json`
  - `ENERGY_OPTIMIZATION_REPORT.json`
  - `thorough_scan_results.json`

