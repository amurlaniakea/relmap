# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""Propagación de incertidumbre por Monte Carlo sobre masas y posiciones estimadas."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .weakfield import trace_ray


@dataclass
class MCStats:
    n: int
    hits: int
    deflection_arcsec: dict
    delay_s: dict


def _summ(a):
    a = np.asarray(a, float)
    if a.size == 0:
        return {k: float("nan") for k in ("mean", "std", "p5", "p50", "p95")}
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "p5": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95))}


def monte_carlo(bodies, ray, n=200, seed=0):
    rng = np.random.default_rng(seed)
    defl, delay, hits = [], [], 0
    for _ in range(n):
        sampled = [replace(
            b, mass_kg=max(b.mass_kg * (1.0 + b.mass_sigma_rel * rng.standard_normal()), 0.0),
            position=b.position + b.position_sigma * rng.standard_normal(3)) for b in bodies]
        r = trace_ray(sampled, ray.origin, ray.direction, ray.length, name=ray.name)
        if r.hit:
            hits += 1
            continue
        defl.append(r.deflection_arcsec)
        delay.append(r.delay_s)
    return MCStats(n=n, hits=hits, deflection_arcsec=_summ(defl), delay_s=_summ(delay))
