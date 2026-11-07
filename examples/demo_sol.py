# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""Demo: luz rozando el Sol + órbita relativista sobre la malla de Flamm."""
from relmap import integrate_null, integrate_timelike
from relmap.viz import build_map

M = 1.0                       # unidades geométricas (M = 1)
luz = [(f"fotón b={b}M", integrate_null(M, r0=200.0, b=b)[0]) for b in (4, 6, 9, 14)]
orb, _ = integrate_timelike(M, 30.0, 60.0, n_orbits=3)
build_map(M, 120.0, luz + [("masa de prueba", orb)],
          title="Schwarzschild: geodésicas sobre la malla de Flamm", out_html="mapa.html")
print("Generado mapa.html")
