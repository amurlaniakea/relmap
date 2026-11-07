# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""relmap: navegación relativista y mapa del espacio-tiempo (Fase 1: Schwarzschild)."""

from .units import G, C, schwarzschild_mass_length, to_geometric_distance
from .geodesics import integrate_null, integrate_timelike, orbit_energy_momentum
from .scene import Body, Ray, Scene, load_scene, parse_scene
from .weakfield import potential, time_dilation, trace_ray
from .uncertainty import monte_carlo
from .solar_system import solar_system_bodies, default_rays, positions, scene_to_dict

__all__ = [
    "G", "C", "schwarzschild_mass_length", "to_geometric_distance",
    "integrate_null", "integrate_timelike", "orbit_energy_momentum",
    "Body", "Ray", "Scene", "load_scene", "parse_scene",
    "potential", "time_dilation", "trace_ray", "monte_carlo",
    "solar_system_bodies", "default_rays", "positions", "scene_to_dict",
]
__version__ = "0.1.0"
