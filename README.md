# LQG-Enhanced Warp Bubble Verification Framework

**Verification framework for LQG-enhanced warp bubble energy optimizations with reproducible computational methods and stability analysis.**

[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](http://unlicense.org/)

## Overview

This repository provides the computational verification framework for the paper *"Verification of LQG Warp Bubble Optimizations: Computational Methods and Limitations"*. It contains tools to:

1.  **Reproduce Energy Optimization Claims**: Separation of "pipeline feasibility ratio" (~30x) from system-level accounting (~1083x).
2.  **Verify Enhancement Factors**: Symbolic derivations and numerical validation for Cavity QED, Squeezing, and LQG methods.
3.  **Analyze Stability**: Iterative backreaction solvers with adaptive damping, 3+1D toy evolution models, and causality screening.
4.  **Explore Parameter Space**: Monte Carlo sensitivity analysis and parameter sweeps.

**Disclaimer**: This work identifies strictly numerical parameter regimes where energy ratios approach \(\mathcal{O}(1)\) *contingent on* heuristic models. It does **not** constitute proof of physical feasibility or existence of a warp drive.

## Key Features

*   **Iterative Backreaction**: Solves Einstein equations with stress-energy feedback (`scripts/backreaction_iterative_experiment.py`).
*   **Enhancement Derivations**: SymPy-based verification of $F_{cav} = \sqrt{Q}$, $F_{sq} = e^r$, etc. (`scripts/derive_enhancements.py`).
*   **Stability Probes**:
    *   Adaptive Damping for solver convergence.
    *   3+1D Toy Evolution (ADM-like) showing stability ($\lambda < 0$).
    *   Curved-Spacetime Quantum Inequality checks (Toy models).
*   **Batch Reproducibility**: Orchestrates complete verification pipeline (`scripts/batch_analysis.py`).

## Installation

```bash
git clone https://github.com/arcticoder/warp-bubble-qft.git
cd warp-bubble-qft
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

The primary interface is the consolidated CLI `scripts/main.py` or the batch runner `scripts/batch_analysis.py`.

### Quick Reproducibility Check
Run the core verification pipeline (approx. 2 minutes):
```bash
python scripts/main.py batch --session-name quick_check --quick
```

### Full Paper Reproduction
Generate all plots and data tables used in the manuscript:
```bash
python scripts/batch_analysis.py --session-name final_verif \
    --include-derivations \
    --include-integrated-qi-3d \
    --use-adaptive-damping
```
Results will be stored in `results/final_verif/`.

### Individual Modules

**Iterative Backreaction**:
```bash
python scripts/backreaction_iterative_experiment.py --mu 0.10 --R 2.3 --outer-iters 5 --save-plots
```

**Toy Evolution**:
```bash
python scripts/toy_evolution.py --mu 0.15 --R 3.0 --t-final 2.0 --save-plots
```

## Repository Structure

*   `papers/`: Manuscript source (`lqg_warp_verification_methods.tex`) and figures.
*   `scripts/`: All executable scripts (CLI, batch runners, experiments).
*   `src/warp_qft/`: Core library code (physics models, solvers).
*   `results/`: Output directory for reproducibility artifacts.
*   `docs/`: Methodological documentation and reports.

## Methods & Verification

Details of the implementation are available in the [manuscript](papers/lqg_warp_verification_methods.pdf) and `docs/` folder.

*   **Energy Discrepancy**: The "1083x" figure from related work refers to a cross-repository total energy accounting. The *pipeline* optimization factor is ~340x (combined) / ~30x (rigorous geometric). See `scripts/discrepancy_analysis.py`.
*   **Quantum Inequalities**: We verify that while flat-space bounds are violated, toy curved-space bounds are satisfied in the optimized regime. See `scripts/curved_qi_verification.py`.

## License

This project is released under the [Unlicense](LICENSE) - explicit dedication to the public domain.
