"""Coarse causality/CTC screening helpers.

These checks are intentionally conservative and *not* a substitute for a full
3+1 GR evolution with constraint monitoring. They are meant to provide early
warnings and reproducible diagnostics.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def screen_spherical_metric(r: np.ndarray, g_tt: np.ndarray, g_rr: np.ndarray) -> Dict:
    """Screen a simple spherically-symmetric metric for obvious pathologies.

    Assumes a diagonal (t, r, ...) metric with components g_tt(r), g_rr(r).

    Flags:
      - signature_violations: g_tt >= 0 or g_rr <= 0
      - nonfinite: NaNs/Infs in metric
      - null_slope_nonfinite: dt/dr for radial null geodesics non-finite

    Returns JSON-safe scalars only.
    """
    r = np.asarray(r)
    g_tt = np.asarray(g_tt)
    g_rr = np.asarray(g_rr)

    nonfinite = (not np.all(np.isfinite(g_tt))) or (not np.all(np.isfinite(g_rr)))

    signature_violation_tt = bool(np.any(g_tt >= 0))
    signature_violation_rr = bool(np.any(g_rr <= 0))

    # Radial null condition: 0 = g_tt dt^2 + g_rr dr^2 => dt/dr = sqrt(g_rr / -g_tt)
    with np.errstate(divide="ignore", invalid="ignore"):
        null_slope = np.sqrt(g_rr / (-g_tt))

    null_slope_nonfinite = bool(np.any(~np.isfinite(null_slope)))

    summary = {
        "nonfinite": bool(nonfinite),
        "signature_violation_tt": signature_violation_tt,
        "signature_violation_rr": signature_violation_rr,
        "null_slope_nonfinite": null_slope_nonfinite,
        "g_tt_min": float(np.min(g_tt)) if g_tt.size else float("nan"),
        "g_tt_max": float(np.max(g_tt)) if g_tt.size else float("nan"),
        "g_rr_min": float(np.min(g_rr)) if g_rr.size else float("nan"),
        "g_rr_max": float(np.max(g_rr)) if g_rr.size else float("nan"),
        "null_slope_min": float(np.nanmin(null_slope)) if null_slope.size else float("nan"),
        "null_slope_max": float(np.nanmax(null_slope)) if null_slope.size else float("nan"),
    }

    summary["ok"] = not (
        summary["nonfinite"]
        or summary["signature_violation_tt"]
        or summary["signature_violation_rr"]
        or summary["null_slope_nonfinite"]
    )

    return summary
