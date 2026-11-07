# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""Mapa interactivo (HTML autocontenido con Plotly)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .embedding import flamm_grid, lift_to_surface


def build_map(M, r_max, geodesics=(), title="relmap", out_html=None):
    """geodesics: iterable de (nombre, sol) con sol.y[1]=r, sol.y[2]=phi."""
    radial, rings = flamm_grid(M, r_max)
    fig = go.Figure()
    for x, y, z in radial + rings:
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", showlegend=False,
                                   line=dict(color="rgba(120,140,200,0.35)", width=1),
                                   hoverinfo="skip"))
    for name, sol in geodesics:
        r, phi = sol.y[1], sol.y[2]
        mask = (r > 2.0 * M * 1.0001) & (r <= r_max)
        x, y, z = lift_to_surface(r[mask], phi[mask], M)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", name=name,
                                   line=dict(width=5)))
    fig.update_layout(title=title, scene=dict(aspectmode="data",
                      xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z embebido (m)"))
    if out_html:
        fig.write_html(out_html, include_plotlyjs="cdn")
    return fig
