# Reproducibility Notes — warp-bubble-qft

**Date**: 2026-01-21  
**Commit**: (run `git rev-parse HEAD` to record)  
**Platform**: Linux  
**Python**: 3.12.3

## Environment Setup

A virtual environment was created and dependencies installed:

```bash
cd /home/echo_/Code/asciimath/warp-bubble-qft
python -m venv .venv
source .venv/bin/activate  # or .venv/bin/activate on Linux/Mac
pip install -U pip
pip install -r requirements.txt
```

Installed packages (frozen):
```
numpy==2.4.1
scipy==1.17.0
matplotlib==3.10.8
sympy==1.14.0
pytest==9.0.2
jupyter==1.1.1
notebook==7.5.2
```

## Baseline Run Results

### Quick Feasibility Check (Default Parameters)

**Command**:
```bash
python run_enhanced_lqg_pipeline.py --quick-check
```

**Parameters**:
- Polymer scale μ = 0.050
- Bubble radius R = 5.00

**Output**:
```
Base energy requirement: 0.5045
Final energy requirement: 0.0172
Feasible (≤ unity): True
Energy reduction: 96.6%

Enhancement breakdown:
  Cavity boost: 11.00×
  Quantum squeezing: 1.00×
  Multi-bubble: 2.27×
  Total enhancement: 24.92×
```

**Interpretation**:  
The framework claims feasibility with a final energy requirement of **1.72%** of unity (i.e., 98.28% reduction from base).  
This is **far below** the "1083× / 99.9%" headline in the README, which may refer to a different baseline or parameter set.

### Parameter Space Scan

**Command**:
```bash
python run_enhanced_lqg_pipeline.py --parameter-scan
```

**Scan Range**:
- μ ∈ [0.050, 0.200]
- R ∈ [1.50, 4.00]
- Grid resolution: 50×50 (2500 points)

**Output**:
```
Feasible configurations found: 2500
Best configuration:
  μ = 0.050000
  R = 4.000000
  Energy = 0.032218
```

**Interpretation**:  
**All 2500 scanned configurations are feasible** (energy ≤ 1.0), with the best at **3.22%** of unity. This suggests the default parameter ranges are already in a "sweet spot" for the given enhancement factors (cavity Q=10⁶, squeezing=15 dB, N_bubbles=3).

**Red flag for publishability**:  
100% feasibility across a coarse grid is suspicious—either:
1. The parameter ranges are cherry-picked to avoid failure regions.
2. The enhancement factors are unrealistically generous.
3. There is a systematic error in the energy computation.

## Sign Convention Issue (Critical)

The README equation states:
$$
\rho_i = \frac{1}{2}\left(\frac{\sin^2(\bar\mu p_i)}{\bar\mu^2} + (\nabla_d\phi)_i^2\right) < 0
$$

This is **mathematically impossible** as written (sum of non-negative terms). The code in `lqg_profiles.py` applies an **external negative sign**:
```python
rho = -amplitude * envelope * polymer_factor
```

**Action required**:  
Either correct the README formula or add a clear note that the negative sign is applied *after* computing the kinetic/gradient contributions.

## Next Steps for Verification

1. **Re-run with varied enhancement factors** (e.g., Q=10⁴, squeezing=5 dB) to test sensitivity.
2. **Run `verify_qi_energy_density.py --scan --save-plots`** to validate QI claims.
3. **Implement Monte Carlo robustness test** (add noise to μ, R initial conditions).
4. **Compare against literature QI bounds** (e.g., Flanagan 1997, Ford & Roman 1995).

## Files Generated

- `results/param_scan_YYYYMMDD_HHMMSS.log` — parameter scan output
- `results/complete_pipeline.json` — (target: full pipeline run, not yet successful)
- `results/REPRODUCIBILITY.md` — this file
