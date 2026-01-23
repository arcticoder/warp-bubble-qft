#!/usr/bin/env python3
"""Compile the REVTeX manuscript under papers/.

This is a small helper to build the canonical manuscript without relying
on shell-specific Makefile behavior.

By default it targets:
  papers/lqg_warp_verification_methods.tex

Usage:
  python compile_manuscript.py
  python compile_manuscript.py --tex lqg_warp_verification_methods.tex
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent
PAPERS_DIR = REPO_ROOT / "papers"


def _run(cmd: List[str], *, cwd: Path) -> int:
    print(f"▸ {cwd}$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(cwd))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--tex",
        type=str,
        default="lqg_warp_verification_methods.tex",
        help="TeX filename inside papers/ (default: lqg_warp_verification_methods.tex)",
    )
    p.add_argument("--clean", action="store_true", help="Remove common LaTeX aux files in papers/ before building")
    args = p.parse_args()

    if not PAPERS_DIR.exists():
        print(f"papers/ directory not found at: {PAPERS_DIR}")
        return 2

    tex_path = Path(args.tex)
    if tex_path.is_absolute():
        # Allow absolute path, but normalize to papers/ to keep outputs co-located.
        if tex_path.parent != PAPERS_DIR:
            print("For reproducibility, pass a filename inside papers/ (or run from papers/ directly).")
            return 2
        tex_file = tex_path.name
    else:
        tex_file = tex_path.name

    full_tex = PAPERS_DIR / tex_file
    if not full_tex.exists():
        print(f"TeX file not found: {full_tex}")
        return 2

    stem = full_tex.stem

    if args.clean:
        for ext in ["aux", "log", "blg", "out", "bbl", "toc"]:
            pth = PAPERS_DIR / f"{stem}.{ext}"
            if pth.exists():
                pth.unlink()

    pdflatex = shutil.which("pdflatex")
    bibtex = shutil.which("bibtex")

    if pdflatex is None:
        make = shutil.which("make")
        if make is None:
            print("Neither pdflatex nor make is available in PATH.")
            return 2
        # Fall back to Makefile (may not halt on errors depending on target).
        return _run([make, "manuscript"], cwd=REPO_ROOT)

    latex_cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_file]

    rc = _run(latex_cmd, cwd=PAPERS_DIR)
    if rc != 0:
        return rc

    # Run BibTeX only if available; REVTeX typically uses it.
    if bibtex is not None:
        _run([bibtex, stem], cwd=PAPERS_DIR)

    # Two more passes for refs/citations.
    rc = _run(latex_cmd, cwd=PAPERS_DIR)
    if rc != 0:
        return rc
    rc = _run(latex_cmd, cwd=PAPERS_DIR)
    if rc != 0:
        return rc

    pdf_path = PAPERS_DIR / f"{stem}.pdf"
    if not pdf_path.exists():
        print(f"Build completed but PDF not found: {pdf_path}")
        return 2

    print(f"✓ Built {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
