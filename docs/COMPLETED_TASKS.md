# Completed Tasks Summary (2026-01-21)

## Overview

Updated TODO.md priorities and implemented the top-priority items for publishable-grade verification:

1. **Iterative/nonlinear backreaction coupling**
2. **Toy 1D evolution harness**
3. **Causality/metric-signature screening**
4. **Discrepancy analysis** (1083× reconciliation)

All new scripts support `--save-results` / `--save-plots` with configurable `--results-dir` for reproducible archiving.

---

## Implemented Components

### 1. Iterative Backreaction ([src/warp_qft/backreaction_solver.py](src/warp_qft/backreaction_solver.py))

- **Function**: `apply_backreaction_correction_iterative(...)`
- **Coupling**: Outer loop scales stress-energy amplitude by current energy estimate; inner loop solves simplified Einstein equations
- **Convergence**: Relative energy tolerance (default 1e-4)
- **Standalone runner**: `backreaction_iterative_experiment.py`
  ```bash
  python backreaction_iterative_experiment.py --save-results --save-plots
  ```
- **Pipeline integration**:
  - Config flags: `backreaction_iterative`, `backreaction_outer_iterations`, `backreaction_relative_energy_tolerance`
  - CLI flags: `--backreaction-iterative`, `--backreaction-outer-iters`, `--backreaction-rel-tol`
  ```bash
  python run_enhanced_lqg_pipeline.py --quick-check --backreaction-iterative
  ```

### 2. Toy Evolution ([toy_evolution.py](toy_evolution.py))

- **Model**: 1D reaction-diffusion PDE driven by polymer energy density profile
- **Output**: Timestamped JSON + plots of metric components
- **Explicit non-claims**:
  - Not constrained 3+1 GR evolution
  - No gauge conditions or constraint damping
  - Stress-energy closure is heuristic
- **Usage**:
  ```bash
  python toy_evolution.py --mu 0.15 --R 3.0 --save-results --save-plots
  ```

### 3. Causality Screening ([src/warp_qft/causality.py](src/warp_qft/causality.py), [causality_screen.py](causality_screen.py))

- **Checks**:
  - Signature violations (g_tt >= 0, g_rr <= 0)
  - Nonfinite values (NaN/Inf)
  - Null-geodesic slopes
- **Integrated**: `toy_evolution.py` automatically screens final metric
- **Standalone**: Screen saved artifacts
  ```bash
  python causality_screen.py results/toy_evolution_*.json --save-results
  ```

### 4. Discrepancy Analysis ([discrepancy_analysis.py](discrepancy_analysis.py))

- **Purpose**: Explicitly separate pipeline "dimensionless feasibility ratio" from ENERGY_OPTIMIZATION_REPORT.json "computational energy accounting"
- **Output**: JSON with interpretation note that these are not the same physical quantity
- **Usage**:
  ```bash
  python discrepancy_analysis.py --save-results
  ```

### 5. Batch Analysis Runner ([batch_analysis.py](batch_analysis.py))

- **Orchestrates**: Quick-check (baseline + iterative), QI scan, sensitivity, toy evolution, discrepancy
- **Output**: Session directory with manifest
- **Usage**:
  ```bash
  python batch_analysis.py --session-name my_session --skip-slow
  ```

---

## Artifacts Generated

All scripts now support `--results-dir <path>` for organized archiving:

```
results/
  final_check/                                  # Example batch session
    qi_scan.png
    energy_density_profile.png
    monte_carlo_feasibility.png
    cavity_Q_sensitivity.png
    squeezing_sensitivity.png
    sensitivity_analysis_*.json
    toy_evolution_*.json
    toy_evolution_*.png
    discrepancy_*.json
    session_manifest.txt
  
  backreaction_iterative_*.json
  backreaction_iterative_*.png
  causality_screen_*.json
  ...
```

---

## TODO.md Status

Updated [docs/TODO.md](docs/TODO.md):

- ✅ Task 1.1: Discrepancy analysis script implemented
- ✅ Task 3: Iterative backreaction mode implemented and integrated
- ✅ Task 4: Toy evolution harness implemented
- ✅ Task 5: Causality screening implemented

Remaining priority work:

- [ ] Task 1.1: Re-run pipeline with toggled baselines to isolate missing factors
- [ ] Task 1.2: Tighten Ford-Roman comparison (explicit mapping to literature)
- [ ] Task 6: Literature benchmarking table

---

## Quick Reference

### Run a coordinated verification batch:
```bash
python batch_analysis.py --session-name verification_$(date +%Y%m%d)
```

### Individual scripts:
```bash
# Iterative backreaction
python backreaction_iterative_experiment.py --mu 0.10 --R 2.3 --outer-iters 5 --save-results --save-plots

# Toy evolution
python toy_evolution.py --mu 0.15 --R 3.0 --t-final 2.0 --save-results --save-plots

# Causality screen
python causality_screen.py results/toy_evolution_*.json --save-results

# Discrepancy report
python discrepancy_analysis.py --save-results

# QI verification
python verify_qi_energy_density.py --scan --save-plots --results-dir results/

# Sensitivity analysis
python sensitivity_analysis.py --trials 100 --save-results --save-plots --results-dir results/
```

### Pipeline with iterative backreaction:
```bash
python run_enhanced_lqg_pipeline.py --quick-check --backreaction-iterative --backreaction-outer-iters 5
```

---

## Testing

- Core pipeline tests: `pytest tests/test_enhancement_pipeline.py -q` (18 passed, 6 pre-existing failures in LQG profile tests)
- New iterative backreaction test: Added mocked unit test to verify plumbing
- All new scripts compile cleanly and run successfully in batch mode

---

## Next Steps (from TODO.md)

1. **Baseline toggles** (Task 1.1 continuation):
   - Run pipeline with/without VdB-Natário to measure geometric contribution
   - Compare quick vs full backreaction solve
   - Sweep enhancement parameters (low Q, low squeezing, N=1)

2. **Literature mapping** (Task 1.2):
   - Document Ford-Roman sampling function choice
   - Add explicit symbol mapping to literature

3. **Benchmarking table** (Task 6):
   - Comparison table: bounds, averaging procedures, parameter mappings
   - Paper-style narrative with known objections and limitations
