# Warp Bubble QFT

This repository contains the implementation of a quantum field theory on a polymer/loop background, focusing on generating stable negative-energy densities (warp bubbles) that can remain in violation of the Ford–Roman quantum inequalities.

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

## Theory Overview

This work extends Loop Quantum Gravity (LQG) to include matter fields quantized on a discrete polymer background. The key innovation is that the discrete nature of the polymer representation allows for:

- **Stable Negative Energy**: The polymer commutation relations modify the uncertainty principle, permitting longer-lived negative energy densities.
- **Warp Bubble Formation**: These negative energy regions can be configured to create stable warp bubble geometries.
- **Ford-Roman Violation**: The discrete field algebra allows violation of classical quantum inequalities over extended time periods.

## License

MIT License
