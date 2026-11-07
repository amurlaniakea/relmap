# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""Escena: cuerpos y rayos leídos de YAML/JSON. Todo se convierte a metros y kg."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .units import G, C

DIST = {"m": 1.0, "km": 1e3, "au": 1.495978707e11, "ly": 9.4607304725808e15,
        "pc": 3.0856775814913673e16, "rsun": 6.957e8}
MASS = {"kg": 1.0, "msun": 1.98847e30, "mearth": 5.9722e24, "mjup": 1.89813e27}


@dataclass(frozen=True)
class Body:
    name: str
    mass_kg: float
    position: np.ndarray
    radius: float = 0.0               # m (0 = puntual)
    mass_sigma_rel: float = 0.0       # incertidumbre relativa de la masa
    position_sigma: float = 0.0       # m, incertidumbre isótropa de la posición
    color: str | None = None          # color opcional en el mapa (#rrggbb)

    @property
    def mass_geo(self) -> float:
        """G m / c^2 en metros."""
        return G * self.mass_kg / C**2


@dataclass(frozen=True)
class Ray:
    name: str
    origin: np.ndarray
    direction: np.ndarray
    length: float


@dataclass
class Scene:
    bodies: list[Body] = field(default_factory=list)
    rays: list[Ray] = field(default_factory=list)


def _vec(v, k):
    a = np.asarray(v, dtype=float) * k
    if a.shape != (3,):
        raise ValueError(f"se esperaba un vector de 3 componentes, llegó {v!r}")
    return a


def parse_scene(data: dict) -> Scene:
    unit = str(data.get("unit", "m")).lower()
    if unit not in DIST:
        raise ValueError(f"unidad de distancia desconocida: {unit} (usa {sorted(DIST)})")
    k = DIST[unit]
    scene = Scene()
    for i, b in enumerate(data.get("bodies", [])):
        mu = str(b.get("mass_unit", "kg")).lower()
        if mu not in MASS:
            raise ValueError(f"unidad de masa desconocida: {mu} (usa {sorted(MASS)})")
        mass = float(b["mass"]) * MASS[mu]
        if mass < 0:
            raise ValueError("la masa no puede ser negativa")
        scene.bodies.append(Body(
            name=str(b.get("name", f"cuerpo{i}")), mass_kg=mass,
            position=_vec(b["position"], k), radius=float(b.get("radius", 0.0)) * k,
            mass_sigma_rel=float(b.get("mass_sigma_rel", 0.0)),
            position_sigma=float(b.get("position_sigma", 0.0)) * k,
            color=b.get("color")))
    for i, r in enumerate(data.get("rays", [])):
        d = _vec(r["direction"], 1.0)
        if np.linalg.norm(d) == 0:
            raise ValueError("la dirección del rayo no puede ser nula")
        scene.rays.append(Ray(name=str(r.get("name", f"rayo{i}")),
                              origin=_vec(r["origin"], k), direction=d / np.linalg.norm(d),
                              length=float(r["length"]) * k))
    return scene


def load_scene(path) -> Scene:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("el archivo de escena debe ser un mapa YAML/JSON")
    return parse_scene(data)
