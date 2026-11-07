# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""Unidades. Internamente G = c = 1 y todo se mide en longitudes (metros)."""

G = 6.67430e-11      # m^3 kg^-1 s^-2
C = 299_792_458.0    # m/s


def schwarzschild_mass_length(mass_kg: float) -> float:
    """Masa en unidades geométricas: M = G m / c^2 (metros)."""
    return G * mass_kg / C**2


def to_geometric_distance(meters: float) -> float:
    return float(meters)
