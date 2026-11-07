# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""Diagrama de incrustación de Flamm: la 'malla deformada' de la geometría espacial.

z(r) = 2 sqrt(2M (r - 2M)) para r >= 2M (corte ecuatorial de Schwarzschild).
"""
from __future__ import annotations

import numpy as np


def flamm_z(r, M):
    r = np.asarray(r, dtype=float)
    return 2.0 * np.sqrt(2.0 * M * np.clip(r - 2.0 * M, 0.0, None))


def flamm_grid(M, r_max, n_r=40, n_phi=48, r_min_factor=2.0001):
    """Rejilla de líneas radiales y circulares sobre el paraboloide de Flamm.

    Devuelve dos listas de polilíneas (x, y, z): radiales y circulares.
    """
    rs = np.linspace(r_min_factor * M, r_max, 400)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    radial = [(rs * np.cos(p), rs * np.sin(p), flamm_z(rs, M)) for p in phis]
    ring_r = np.linspace(r_min_factor * M, r_max, n_r)
    th = np.linspace(0, 2 * np.pi, 361)
    rings = [(r * np.cos(th), r * np.sin(th), np.full_like(th, flamm_z(r, M))) for r in ring_r]
    return radial, rings


def lift_to_surface(r, phi, M):
    """Proyecta puntos (r, phi) de una geodésica sobre el paraboloide."""
    r = np.asarray(r)
    return r * np.cos(phi), r * np.sin(phi), flamm_z(r, M)
