#!/usr/bin/env python3
"""Visualization helpers for stress-test robustness outputs.

Reads a `stress_tests_*.json` artifact produced by `stress_test.py` and renders:
- Robustness bar chart: D per edge case (color-coded robust vs fragile)

Usage:
  python visualize_robustness.py results/stress_tests_*.json --out results/robustness_summary.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract(report: Dict[str, Any]) -> List[Tuple[str, float, str]]:
    out: List[Tuple[str, float, str]] = []
    for item in report.get("edge_cases", []):
        label = str(item.get("label", "unlabeled"))
        stats = item.get("final_energy_stats", {})
        D = float(stats.get("robustness_D", float("nan")))
        interp = str(item.get("interpretation", ""))
        out.append((label, D, interp))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=str, help="Path to stress_tests_*.json")
    ap.add_argument("--out", type=str, default="robustness_summary.png", help="Output PNG path")
    ap.add_argument("--title", type=str, default="Stress-Test Robustness Summary", help="Plot title")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}")
        return 2

    rows = _extract(_load_report(in_path))
    if not rows:
        print("No edge_cases found in report.")
        return 2

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting (pip install matplotlib)")
        return 2

    labels = [r[0] for r in rows]
    D_vals = [r[1] for r in rows]
    colors = ["tab:red" if r[2] == "fragile" else "tab:green" for r in rows]

    fig_h = max(5.0, 0.35 * len(labels))
    fig, ax = plt.subplots(figsize=(10, fig_h))

    y = list(range(len(labels)))
    ax.barh(y, D_vals, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Robustness D = std(E)/mean(E)")
    ax.set_title(args.title)
    ax.axvline(0.1, color="gray", linestyle=":", alpha=0.7, linewidth=1.5, label="D=0.1 threshold")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
