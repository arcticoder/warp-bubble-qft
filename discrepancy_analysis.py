#!/usr/bin/env python3
"""Discrepancy analysis: pipeline energy ratio vs 1083× report.

This script produces a single JSON artifact that makes it explicit that:
- The enhancement pipeline reports a *dimensionless feasibility energy ratio*
- ENERGY_OPTIMIZATION_REPORT.json reports a *computational energy accounting*

It is meant to reduce ambiguity when discussing the headline “1083× / 99.9%”.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
    except Exception:
        pass

    return str(obj)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mu", type=float, default=0.10)
    p.add_argument("--R", type=float, default=2.3)
    p.add_argument("--report", type=str, default="ENERGY_OPTIMIZATION_REPORT.json")

    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--save-results", action="store_true")

    args = p.parse_args()

    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    # Import lazily so the script can still run just to summarize the report.
    from src.warp_qft.enhancement_pipeline import run_quick_feasibility_check

    pipeline = run_quick_feasibility_check(mu=float(args.mu), R=float(args.R))

    out: Dict[str, Any] = {
        "timestamp": _timestamp(),
        "pipeline": {
            "parameters": pipeline.get("parameters"),
            "base_energy_ratio": pipeline.get("base_energy"),
            "final_energy_ratio": pipeline.get("final_energy"),
            "feasible": pipeline.get("feasible"),
        },
        "energy_optimization_report": {
            "path": str(report_path),
            "achieved_optimization_factor": report.get("achieved_optimization_factor"),
            "energy_metrics": report.get("energy_metrics"),
            "system_optimization_results": report.get("system_optimization_results"),
        },
        "interpretation": {
            "pipeline_quantity": "dimensionless feasibility ratio (target <= 1)",
            "report_quantity": "computational energy accounting (J/GJ/MJ) from cross-repository integration",
            "note": "These are not the same physical quantity; do not compare ratios directly without a mapping.",
        },
    }

    if args.save_results:
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"discrepancy_{out['timestamp']}.json"
        out_path.write_text(json.dumps(_json_safe(out), indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    print(json.dumps(_json_safe(out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
