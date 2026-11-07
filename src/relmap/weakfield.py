# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""Campo débil estático multi-cuerpo.

Métrica isótropa: ds^2 = -(1+2Phi) dt^2 + (1-2Phi) dx^2, con Phi = -sum_i m_i / r_i
(m_i = G M_i / c^2 en metros, G = c = 1). Válida para |Phi| << 1, cuerpos lentos.
La luz se propaga como en un medio con índice n = 1 - 2 Phi:  d/ds (n dx/ds) = grad n.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .units import C


def potential(points, bodies):
    """Phi (adimensional) en uno o varios puntos. Esfera uniforme dentro del radio."""
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    pts = np.atleast_2d(pts)
    phi = np.zeros(len(pts))
    for b in bodies:
        r = np.linalg.norm(pts - b.position, axis=1)
        m = b.mass_geo
        if b.radius > 0:
            out = -m / np.where(r < b.radius, 1.0, r)
            inn = -m * (3 * b.radius**2 - r**2) / (2 * b.radius**3)
            phi += np.where(r < b.radius, inn, out)
        else:
            phi += -m / np.maximum(r, 1e-30)
    return float(phi[0]) if single else phi


def grad_potential(x, bodies):
    g = np.zeros(3)
    for b in bodies:
        d = x - b.position
        r = float(np.linalg.norm(d))
        m = b.mass_geo
        if b.radius > 0 and r < b.radius:
            g += m * d / b.radius**3
        else:
            g += m * d / max(r, 1e-30) ** 3
    return g


def time_dilation(point, bodies):
    """dtau/dt de un observador estático: sqrt(1 + 2 Phi)."""
    return float(np.sqrt(1.0 + 2.0 * potential(point, bodies)))


@dataclass
class RayResult:
    name: str
    sol: object
    s_end: float
    deflection_rad: float
    delay_m: float          # integral de (n-1) ds (retraso tipo Shapiro) en metros
    hit: str | None

    @property
    def deflection_arcsec(self) -> float:
        return float(np.degrees(self.deflection_rad) * 3600.0)

    @property
    def delay_s(self) -> float:
        return self.delay_m / C

    def path(self, n=300):
        s = np.linspace(0.0, self.s_end, n)
        return self.sol.sol(s)[:3].T


def trace_ray(bodies, origin, direction, length, name="rayo", rtol=1e-12):
    origin = np.asarray(origin, float)
    d0 = np.asarray(direction, float)
    d0 = d0 / np.linalg.norm(d0)
    n0 = 1.0 - 2.0 * potential(origin, bodies)
    y0 = np.concatenate([origin, n0 * d0, [0.0]])

    def rhs(_s, y):
        x, p = y[:3], y[3:6]
        n = 1.0 - 2.0 * potential(x, bodies)
        return np.concatenate([p / n, -2.0 * grad_potential(x, bodies), [n - 1.0]])

    def make_event(b):
        def ev(_s, y):
            return float(np.linalg.norm(y[:3] - b.position) - b.radius)
        ev.terminal = True
        ev.direction = -1
        return ev

    solid = [b for b in bodies if b.radius > 0]
    sol = solve_ivp(rhs, (0.0, length), y0, method="DOP853", rtol=rtol,
                    atol=np.array([1e-3] * 3 + [1e-15] * 3 + [1e-6]),
                    events=[make_event(b) for b in solid] or None, dense_output=True)
    hit = None
    for b, te in zip(solid, sol.t_events if solid else []):
        if len(te):
            hit = b.name
    s_end = float(sol.t[-1])
    pf = sol.y[3:6, -1]
    cross = np.linalg.norm(np.cross(d0, pf))
    defl = float(np.arctan2(cross, float(np.dot(d0, pf))))
    return RayResult(name=name, sol=sol, s_end=s_end,
                     deflection_rad=float("nan") if hit else defl,
                     delay_m=float(sol.y[6, -1]), hit=hit)
