#!/usr/bin/env python3
"""Batch analysis runner for reproducibility artifacts.

Runs a coordinated set of verification/analysis scripts and archives outputs
into a single timestamped session directory under results/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_cmd(cmd: List[str], desc: str) -> Tuple[int, str]:
    print(f"▸ {desc}")
    print(f"  {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  ⚠ exit code {result.returncode}")
            print(result.stderr[:500] if result.stderr else "")
        else:
            print(f"  ✓")
        return result.returncode, result.stdout
    except subprocess.TimeoutExpired:
        print(f"  ⚠ timeout (300s)")
        return -1, ""
    except Exception as exc:
        print(f"  ⚠ {exc}")
        return -1, ""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session-name", type=str, default=None, help="Optional session label (default: timestamp)")
    p.add_argument("--skip-slow", action="store_true", help="Skip slow iterative backreaction")
    args = p.parse_args()

    ts = _timestamp()
    session = args.session_name if args.session_name else f"session_{ts}"
    session_dir = Path("results") / session
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"Batch analysis session: {session}")
    print(f"Output directory: {session_dir}")
    print("=" * 60)

    tasks = []

    # Quick feasibility check (baseline)
    tasks.append(
        (
            ["python", "run_enhanced_lqg_pipeline.py", "--quick-check"],
            "Quick feasibility check (baseline)",
        )
    )

    # Quick feasibility with iterative backreaction
    tasks.append(
        (
            [
                "python",
                "run_enhanced_lqg_pipeline.py",
                "--quick-check",
                "--backreaction-iterative",
                "--backreaction-outer-iters",
                "3",
            ],
            "Quick feasibility check (iterative backreaction)",
        )
    )

    # QI verification scan
    tasks.append(
        (
            [
                "python",
                "verify_qi_energy_density.py",
                "--scan",
                "--save-plots",
                "--results-dir",
                str(session_dir),
            ],
            "QI verification scan",
        )
    )
    
    # Sensitivity analysis
    tasks.append(
        (
            [
                "python",
                "sensitivity_analysis.py",
                "--trials",
                "100",
                "--save-results",
                "--save-plots",
                "--results-dir",
                str(session_dir),
            ],
            "Sensitivity analysis (Monte Carlo + enhancement sweeps)",
        )
    )

    # Toy evolution
    tasks.append(
        (
            [
                "python",
                "toy_evolution.py",
                "--save-results",
                "--save-plots",
                "--results-dir",
                str(session_dir),
            ],
            "Toy 1D evolution",
        )
    )

    # Discrepancy analysis
    tasks.append(
        (
            ["python", "discrepancy_analysis.py", "--save-results", "--results-dir", str(session_dir)],
            "Discrepancy analysis",
        )
    )

    # Baseline comparison (isolate reduction factors)
    tasks.append(
        (
            ["python", "baseline_comparison.py", "--save-results", "--results-dir", str(session_dir)],
            "Baseline comparison (factor isolation)",
        )
    )

    # Iterative backreaction experiment (optional; can be slow)
    if not args.skip_slow:
        tasks.append(
            (
                [
                    "python",
                    "backreaction_iterative_experiment.py",
                    "--grid-size",
                    "120",
                    "--inner-iters",
                    "10",
                    "--outer-iters",
                    "3",
                    "--save-results",
                    "--save-plots",
                    "--results-dir",
                    str(session_dir),
                ],
                "Iterative backreaction experiment (reduced resolution)",
            )
        )

    results = []
    for cmd, desc in tasks:
        code, stdout = _run_cmd(cmd, desc)
        results.append((desc, code))

    print("=" * 60)
    print("Summary:")
    for desc, code in results:
        status = "✓" if code == 0 else f"⚠ {code}"
        print(f"  [{status}] {desc}")

    print(f"\nSession outputs in: {session_dir.resolve()}")

    # Write a session manifest
    manifest_path = session_dir / "session_manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(f"Session: {session}\n")
        f.write(f"Timestamp: {ts}\n")
        f.write("Tasks:\n")
        for desc, code in results:
            f.write(f"  [{code}] {desc}\n")

    print(f"Manifest: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
