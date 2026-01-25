# ArXiv Submission Instructions

This directory contains the source files for the manuscript "Verification of LQG Warp Bubble Optimizations: Computational Methods and Limitations".

## Files matched to ArXiv requirements

- **Main TeX file:** `lqg_warp_verification_methods.tex`
- **Bibliography:** `refs.bib`
- **Configuration:** `author_config.tex`
- **Figures:** `figures/` directory containing all PNG images used in the manuscript.

## Compilation

The manuscript is built using `pdflatex` and `bibtex` (standard REVTeX 4.2 workflow).

```bash
pdflatex lqg_warp_verification_methods
bibtex lqg_warp_verification_methods
pdflatex lqg_warp_verification_methods
pdflatex lqg_warp_verification_methods
```

## Notes for Moderators

- This submission relies on standard `revtex4-2` class and standard packages (`amsmath`, `graphicx`, `hyperref`, etc.).
- No non-standard `.sty` files are required.
- All figures are included in the `figures/` subdirectory.
