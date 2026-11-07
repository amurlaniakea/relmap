# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""Geodésicas en el plano ecuatorial de Schwarzschild (G = c = 1).

Coordenadas (t, r, phi), parámetro afín lambda (tiempo propio para geodésicas temporales).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def _rhs(_lam, y, M):
    t, r, phi, td, rd, phid = y
    f = r - 2.0 * M
    tdd = -2.0 * M / (r * f) * td * rd
    rdd = -M * f / r**3 * td**2 + M / (r * f) * rd**2 + f * phid**2
    phidd = -2.0 / r * rd * phid
    return [td, rd, phid, tdd, rdd, phidd]


def _run(y0, M, lam_max, events=None, rtol=1e-12, atol=1e-12, dense=False):
    return solve_ivp(
        _rhs, (0.0, lam_max), y0, args=(M,), method="DOP853",
        rtol=rtol, atol=atol, events=events, dense_output=dense,
    )


def integrate_null(M: float, r0: float, b: float, lam_max: float | None = None):
    """Fotón que llega desde r0 (>> M) con parámetro de impacto b, E = 1.

    Devuelve (sol, deflexion_rad). La deflexión se mide respecto a la recta plana
    que saldría de r0 con el mismo b (corrige el efecto de r0 finito).
    """
    if b >= r0:
        raise ValueError("b debe ser menor que r0")
    f0 = 1.0 - 2.0 * M / r0
    rd0 = -np.sqrt(1.0 - (b**2 / r0**2) * f0)
    y0 = [0.0, r0, 0.0, 1.0 / f0, rd0, b / r0**2]
    lam_max = lam_max if lam_max is not None else 3.0 * r0

    def escaped(_l, y, _M):
        return y[1] - r0 * 1.0000001
    escaped.terminal = True
    escaped.direction = 1

    def captured(_l, y, _M):
        return y[1] - 2.0 * M * 1.0001
    captured.terminal = True
    captured.direction = -1

    sol = _run(y0, M, lam_max, events=[escaped, captured])
    if sol.t_events[1].size:
        return sol, float("nan")  # capturado por el horizonte
    dphi = sol.y[2, -1]
    flat = np.pi - 2.0 * np.arcsin(b / r0)
    return sol, dphi - flat


def orbit_energy_momentum(M: float, r_p: float, r_a: float):
    """E y L (por unidad de masa) de una órbita ligada con periastro r_p y apoastro r_a."""
    A = np.array([[-1.0, 1.0 / r_p**2 - 2.0 * M / r_p**3],
                  [-1.0, 1.0 / r_a**2 - 2.0 * M / r_a**3]])
    rhs = np.array([2.0 * M / r_p, 2.0 * M / r_a])
    e2m1, L2 = np.linalg.solve(A, rhs)
    return float(np.sqrt(1.0 + e2m1)), float(np.sqrt(L2))


def integrate_timelike(M: float, r_p: float, r_a: float, n_orbits: int = 3):
    """Órbita de una masa de prueba desde el periastro. Devuelve (sol, phi_en_periastros)."""
    E, L = orbit_energy_momentum(M, r_p, r_a)
    f = 1.0 - 2.0 * M / r_p
    y0 = [0.0, r_p, 0.0, E / f, 0.0, L / r_p**2]
    a = 0.5 * (r_p + r_a)
    period = 2.0 * np.pi * np.sqrt(a**3 / M)

    def peri(_l, y, _M):
        return y[4]
    peri.direction = 1

    sol = _run(y0, M, (n_orbits + 0.5) * period * 1.3, events=[peri])
    phis = sol.y_events[0][:, 2] if sol.y_events[0].size else np.array([])
    return sol, phis
