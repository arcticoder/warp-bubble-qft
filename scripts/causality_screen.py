#!/usr/bin/env python3
"""Coarse causality/metric-signature screen for saved outputs.

Primarily intended for JSON artifacts produced by toy_evolution.py.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from src.warp_qft.causality import screen_spherical_metric


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
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return str(obj)


def _extract_metric(payload: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # toy_evolution.py layout
    final = payload.get("final")
    if isinstance(final, dict) and all(k in final for k in ("r", "g_tt", "g_rr")):
        r = np.asarray(final["r"], dtype=float)
        g_tt = np.asarray(final["g_tt"], dtype=float)
        g_rr = np.asarray(final["g_rr"], dtype=float)
        return r, g_tt, g_rr

    raise ValueError("Unsupported JSON format: expected keys final/r, final/g_tt, final/g_rr")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="Path to JSON artifact (e.g., results/toy_evolution_*.json)")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--save-results", action="store_true")
    args = p.parse_args()

    inp = Path(args.input)
    payload = json.loads(inp.read_text(encoding="utf-8"))

    r, g_tt, g_rr = _extract_metric(payload)
    screen = screen_spherical_metric(r, g_tt, g_rr)

    out = {
        "timestamp": _timestamp(),
        "input": str(inp),
        "screen": screen,
    }

    print(json.dumps(out, indent=2))

    if args.save_results:
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / f"causality_screen_{out['timestamp']}.json"
        out_path.write_text(json.dumps(_json_safe(out), indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
