# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""relmap: navegación relativista y mapa del espacio-tiempo (Fase 1: Schwarzschild)."""

from .geodesics import integrate_null, integrate_timelike, orbit_energy_momentum
from .scene import Body, Ray, Scene, load_scene, parse_scene
from .solar_system import default_rays, positions, scene_to_dict, solar_system_bodies
from .uncertainty import monte_carlo
from .units import C, G, schwarzschild_mass_length, to_geometric_distance
from .weakfield import potential, time_dilation, trace_ray

__all__ = [
    "Body",
    "C",
    "G",
    "Ray",
    "Scene",
    "default_rays",
    "integrate_null",
    "integrate_timelike",
    "load_scene",
    "monte_carlo",
    "orbit_energy_momentum",
    "parse_scene",
    "positions",
    "potential",
    "scene_to_dict",
    "schwarzschild_mass_length",
    "solar_system_bodies",
    "time_dilation",
    "to_geometric_distance",
    "trace_ray",
]
__version__ = "0.1.0"
