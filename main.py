#!/usr/bin/env python3
"""Convenience entrypoint for common repo workflows.

This repo historically accumulated many top-level runner scripts.
`main.py` provides a stable, discoverable CLI surface while we gradually
move reusable logic into `src/warp_qft/`.

Examples:
  python main.py batch -- --session-name final_integration
  python main.py full-integration -- --save-results --results-dir results/final_integration
  python main.py stress-test -- --trials 50 --save-results --results-dir results/final_integration
  python main.py compile-manuscript
  python main.py demo vdb
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parent


def _run(cmd: List[str], *, cwd: Optional[Path] = None) -> int:
    cwd_str = str(cwd) if cwd is not None else str(REPO_ROOT)
    print(f"▸ {cwd_str}$ {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=str(cwd) if cwd is not None else str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_batch = sub.add_parser("batch", help="Run batch_analysis.py")
    p_batch.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to batch_analysis.py")

    p_full = sub.add_parser("full-integration", help="Run full_integration.py")
    p_full.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to full_integration.py")

    p_stress = sub.add_parser("stress-test", help="Run stress_test.py")
    p_stress.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to stress_test.py")

    p_compile = sub.add_parser("compile-manuscript", help="Build the REVTeX manuscript under papers/")
    p_compile.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to compile_manuscript.py")

    p_demo = sub.add_parser("demo", help="Run demo scripts")
    p_demo.add_argument("which", choices=["vdb", "fast-scanning"], help="Which demo to run")
    p_demo.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the demo script")

    args = parser.parse_args()

    py = sys.executable

    if args.command == "batch":
        return _run([py, "batch_analysis.py", *args.args])

    if args.command == "full-integration":
        return _run([py, "full_integration.py", *args.args])

    if args.command == "stress-test":
        return _run([py, "stress_test.py", *args.args])

    if args.command == "compile-manuscript":
        script = REPO_ROOT / "compile_manuscript.py"
        if not script.exists():
            print("compile_manuscript.py not found; try: make manuscript")
            return 2
        return _run([py, str(script), *args.args])

    if args.command == "demo":
        demos_dir = REPO_ROOT / "demos"
        if args.which == "vdb":
            script = demos_dir / "demo_van_den_broeck_natario.py"
        else:
            script = demos_dir / "demo_fast_scanning.py"

        if not script.exists():
            print(f"Demo script not found: {script}")
            return 2

        return _run([py, str(script), *args.args])

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
