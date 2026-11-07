# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""Catálogo del Sistema Solar: Sol, 8 planetas, Plutón y la Luna, para una fecha dada.

Posiciones: elementos keplerianos aproximados de JPL (E. M. Standish, Tabla 1, válida
1800-2050), marco eclíptico J2000 heliocéntrico. Errores nominales JPL en 1800-2050:
Mercurio ~1000 km, Venus ~4000 km, Tierra ~6000 km, Marte ~25000 km, Júpiter ~600000 km,
Saturno ~1.5 millones km, Urano ~1 millón km, Neptuno ~200000 km.
Masas y radios: JPL "Planetary Physical Parameters" (radio ecuatorial).
La Luna usa un modelo circular simple (error de decenas de miles de km).
Cuenta el Sol en el origen (no el baricentro): es un marco heliocéntrico.
"""
from __future__ import annotations

import unicodedata
from datetime import datetime, timezone

import numpy as np

from .scene import DIST, Body, Ray, Scene

AU = DIST["au"]
KM = 1e3

# a, e, I, L, long.peri, long.node  (valores a J2000)  y sus tasas por siglo
_EL = {
    "mercurio": ([0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593],
                 [0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081]),
    "venus":    ([0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255],
                 [0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418]),
    "embary":   ([1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0],
                 [0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0]),
    "marte":    ([1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891],
                 [0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343]),
    "jupiter":  ([5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909],
                 [-0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106]),
    "saturno":  ([9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448],
                 [-0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794]),
    "urano":    ([19.18916464, 0.04725744, 0.77263783, 313.23810451, 170.95427630, 74.01692503],
                 [-0.00196176, -0.00004397, -0.00242939, 428.48202785, 0.40805281, 0.04240589]),
    "neptuno":  ([30.06992276, 0.00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574],
                 [0.00026291, 0.00005105, 0.00035372, 218.45945325, -0.32241464, -0.00508664]),
    # Plutón: tabla original de Standish (JPL la retiró de su web tras 2006; los valores se
    # reproducen idénticos en pykep y satkit). Sin error nominal publicado: ver _PHYS.
    "pluton":   ([39.48211675, 0.24882730, 17.14001206, 238.92903833, 224.06891629, 110.30393684],
                 [-0.00031596, 0.00005170, 0.00004818, 145.20780515, -0.04062942, -0.01183482]),
}

# nombre, masa (kg), incertidumbre relativa de la masa, radio ecuatorial (km), error nominal posición (km)
_PHYS = {
    "sol":      ("Sol",      1.98847e30,  1e-4,      695700.0,   0.0),
    "mercurio": ("Mercurio", 0.330103e24, 0.000021 / 0.330103, 2440.53,  1000.0),
    "venus":    ("Venus",    4.86731e24,  0.00023 / 4.86731,   6051.8,   4000.0),
    "tierra":   ("Tierra",   5.97217e24,  0.00028 / 5.97217,   6378.1366, 6000.0),
    "luna":     ("Luna",     7.346e22,    1e-3,      1737.4,     40000.0),
    "marte":    ("Marte",    0.641691e24, 0.00003 / 0.641691,  3396.19,  25000.0),
    "jupiter":  ("Júpiter",  1898.125e24, 0.088 / 1898.125,    71492.0,  600000.0),
    "saturno":  ("Saturno",  568.317e24,  0.026 / 568.317,     60268.0,  1500000.0),
    "urano":    ("Urano",    86.8099e24,  0.004 / 86.8099,     25559.0,  1000000.0),
    "neptuno":  ("Neptuno",  102.4092e24, 0.0048 / 102.4092,   24764.0,  200000.0),
    # masa y radio: JPL Planetary Physical Parameters (planeta enano). El error de posición NO está
    # publicado para Plutón: 3e7 km es una estimación gruesa nuestra (≈1000″ a 39 UA).
    "pluton":   ("Plutón",   13024.6e18,  6.0 / 13024.6,       1188.3,   3.0e7),
}
_ORDER = ["sol", "mercurio", "venus", "tierra", "luna", "marte", "jupiter", "saturno", "urano", "neptuno", "pluton"]
_MOON_DIST = 384400.0 * KM
_MOON_INC = np.radians(5.145)


def _key(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()
    return s


def julian_date(when: datetime) -> float:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    ref = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
    return 2451545.0 + (when - ref).total_seconds() / 86400.0


def solve_kepler(M_deg: float, e: float, tol_deg: float = 1e-10) -> float:
    """Resuelve M = E - e* sin E (grados) y devuelve E en grados."""
    estar = np.degrees(e)
    M = (M_deg + 180.0) % 360.0 - 180.0
    E = M + estar * np.sin(np.radians(M))
    for _ in range(100):
        dM = M - (E - estar * np.sin(np.radians(E)))
        dE = dM / (1.0 - e * np.cos(np.radians(E)))
        E += dE
        if abs(dE) <= tol_deg:
            break
    return float(E)


def _helio(key: str, jd: float) -> np.ndarray:
    """Posición heliocéntrica eclíptica J2000 (m) de un planeta (o del baricentro Tierra-Luna)."""
    el0, rate = _EL[key]
    T = (jd - 2451545.0) / 36525.0
    a, e, I, L, wbar, Om = (el0[i] + rate[i] * T for i in range(6))
    omega = wbar - Om
    M = L - wbar
    E = np.radians(solve_kepler(M, e))
    xp = a * (np.cos(E) - e)
    yp = a * np.sqrt(1 - e * e) * np.sin(E)
    co, so = np.cos(np.radians(omega)), np.sin(np.radians(omega))
    cO, sO = np.cos(np.radians(Om)), np.sin(np.radians(Om))
    cI, sI = np.cos(np.radians(I)), np.sin(np.radians(I))
    x = (co * cO - so * sO * cI) * xp + (-so * cO - co * sO * cI) * yp
    y = (co * sO + so * cO * cI) * xp + (-so * sO + co * cO * cI) * yp
    z = (so * sI) * xp + (co * sI) * yp
    return np.array([x, y, z]) * AU


def moon_offset(jd: float) -> np.ndarray:
    """Posición de la Luna respecto a la Tierra (m), modelo circular simple."""
    T = (jd - 2451545.0) / 36525.0
    lam = np.radians((218.3164477 + 481267.88123421 * T) % 360.0)
    node = np.radians((125.0445479 - 1934.1362891 * T) % 360.0)
    u = lam - node
    cu, su, cn, sn, ci, si = np.cos(u), np.sin(u), np.cos(node), np.sin(node), np.cos(_MOON_INC), np.sin(_MOON_INC)
    return _MOON_DIST * np.array([cu * cn - su * sn * ci, cu * sn + su * cn * ci, su * si])


def orbit_path(name: str, when: datetime, n: int = 240):
    """Trayectoria (n,3) en metros de un cuerpo alrededor de su centro (Sol; la Luna, la Tierra)."""
    k = _key(name)
    jd = julian_date(when)
    if k == "luna":
        T = (jd - 2451545.0) / 36525.0
        node = np.radians((125.0445479 - 1934.1362891 * T) % 360.0)
        u = np.linspace(0, 2 * np.pi, n)
        ci, si, cn, sn = np.cos(_MOON_INC), np.sin(_MOON_INC), np.cos(node), np.sin(node)
        rel = _MOON_DIST * np.stack([np.cos(u) * cn - np.sin(u) * sn * ci,
                                     np.cos(u) * sn + np.sin(u) * cn * ci, np.sin(u) * si], axis=1)
        return positions(when)["tierra"] + rel
    key = "embary" if k == "tierra" else k
    if key not in _EL:
        return None
    el0, rate = _EL[key]
    Tc = (jd - 2451545.0) / 36525.0
    a, e, I, _L, wbar, Om = (el0[i] + rate[i] * Tc for i in range(6))
    omega = np.radians(wbar - Om)
    cO, sO, cI, sI = (np.cos(np.radians(Om)), np.sin(np.radians(Om)),
                      np.cos(np.radians(I)), np.sin(np.radians(I)))
    E = np.linspace(0, 2 * np.pi, n)
    xp, yp = a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E)
    co, so = np.cos(omega), np.sin(omega)
    x = (co * cO - so * sO * cI) * xp + (-so * cO - co * sO * cI) * yp
    y = (co * sO + so * cO * cI) * xp + (-so * sO + co * cO * cI) * yp
    z = (so * sI) * xp + (co * sI) * yp
    return np.stack([x, y, z], axis=1) * AU


def positions(when: datetime) -> dict[str, np.ndarray]:
    jd = julian_date(when)
    out = {"sol": np.zeros(3)}
    for k in ("mercurio", "venus", "marte", "jupiter", "saturno", "urano", "neptuno", "pluton"):
        out[k] = _helio(k, jd)
    emb = _helio("embary", jd)
    mm, me = _PHYS["luna"][1], _PHYS["tierra"][1]
    rel = moon_offset(jd)
    out["tierra"] = emb - rel * mm / (me + mm)
    out["luna"] = out["tierra"] + rel
    return out


def solar_system_bodies(when: datetime | None = None, include=None) -> list[Body]:
    when = when or datetime.now(timezone.utc)
    pos = positions(when)
    keys = _ORDER if include is None else [_key(n) for n in include]
    bodies = []
    for k in keys:
        if k not in _PHYS:
            raise ValueError(f"cuerpo desconocido: {k} (disponibles: {_ORDER})")
        name, mass, msig, rkm, psig = _PHYS[k]
        bodies.append(Body(name=name, mass_kg=mass, position=pos[k], radius=rkm * KM,
                           mass_sigma_rel=msig, position_sigma=psig * KM))
    return bodies


def default_rays(bodies, b_factor: float = 2.0, span: float = 2000.0) -> list[Ray]:
    """Un rayo rasante por cuerpo, en el plano eclíptico, pasando a b = b_factor·R."""
    rays = []
    for b in bodies:
        if b.radius <= 0:
            continue
        r = np.array([b.position[0], b.position[1], 0.0])
        u = r / np.linalg.norm(r) if np.linalg.norm(r) > 0 else np.array([1.0, 0, 0])
        t = np.cross([0, 0, 1.0], u)
        t /= np.linalg.norm(t)
        L = span * b.radius
        rays.append(Ray(name=f"rasante {b.name} (b={b_factor:g} R)",
                        origin=b.position + b_factor * b.radius * u - L * t,
                        direction=t, length=2 * L))
    return rays


def scene_to_dict(scene: Scene) -> dict:
    """Exporta a YAML/JSON en UA (editable y reutilizable)."""
    return {
        "unit": "au",
        "bodies": [{"name": b.name, "mass": float(b.mass_kg), "mass_unit": "kg",
                    "position": [float(v / AU) for v in b.position], "radius": float(b.radius / AU),
                    "mass_sigma_rel": float(b.mass_sigma_rel),
                    "position_sigma": float(b.position_sigma / AU)} for b in scene.bodies],
        "rays": [{"name": r.name, "origin": [float(v / AU) for v in r.origin],
                  "direction": [float(v) for v in r.direction], "length": float(r.length / AU)}
                 for r in scene.rays],
    }


# ---------------------------------------------------------------------------------------------
# Cinturón principal de asteroides (ilustrativo): entre ~2,1 y ~3,3 UA, con los huecos de Kirkwood
# ---------------------------------------------------------------------------------------------
BELT_INNER_AU, BELT_OUTER_AU = 2.1, 3.3
_KIRKWOOD_AU = (2.50, 2.82, 2.95, 3.27)          # resonancias 3:1, 5:2, 7:3 y 2:1 con Júpiter


def asteroid_belt(n: int = 1500, seed: int = 7) -> np.ndarray:
    """(n,3) posiciones heliocéntricas (m) de una nube ilustrativa del cinturón principal.

    No son asteroides reales: semiejes con pico en ~2,7 UA y huecos de Kirkwood, excentricidades
    pequeñas e inclinaciones de unos grados, con ángulos al azar. Sirve para ver DÓNDE está el
    cinturón y que separa los planetas interiores de los exteriores.
    """
    rng = np.random.default_rng(seed)
    a = np.empty(0)
    while a.size < n:
        c = rng.normal(2.75, 0.35, size=4 * n)
        c = c[(c > BELT_INNER_AU) & (c < BELT_OUTER_AU)]
        for g in _KIRKWOOD_AU:
            c = c[np.abs(c - g) > 0.025]
        a = np.concatenate([a, c])
    a = a[:n]
    e = np.minimum(rng.rayleigh(0.09, n), 0.35)
    inc = np.radians(np.minimum(rng.rayleigh(7.0, n), 30.0))
    Om, om, M = (rng.uniform(0, 2 * np.pi, n) for _ in range(3))
    E = M.copy()
    for _ in range(30):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    xp, yp = a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E)
    co, so, cO, sO, cI, sI = np.cos(om), np.sin(om), np.cos(Om), np.sin(Om), np.cos(inc), np.sin(inc)
    x = (co * cO - so * sO * cI) * xp + (-so * cO - co * sO * cI) * yp
    y = (co * sO + so * cO * cI) * xp + (-so * sO + co * cO * cI) * yp
    z = (so * sI) * xp + (co * sI) * yp
    return np.stack([x, y, z], axis=1) * AU


def orbit_elements(name: str, when: datetime):
    """(a [UA], e, q [UA], Q [UA]) del cuerpo en esa fecha (la Tierra usa el baricentro Tierra-Luna)."""
    k = _key(name)
    key = "embary" if k == "tierra" else k
    if key not in _EL:
        return None
    el0, rate = _EL[key]
    T = (julian_date(when) - 2451545.0) / 36525.0
    a, e = el0[0] + rate[0] * T, el0[1] + rate[1] * T
    return a, e, a * (1 - e), a * (1 + e)


def apsides(name: str, when: datetime):
    """Posiciones (m) del perihelio y del afelio de la órbita actual: (peri, afe) o None."""
    k = _key(name)
    key = "embary" if k == "tierra" else k
    if key not in _EL:
        return None
    el0, rate = _EL[key]
    T = (julian_date(when) - 2451545.0) / 36525.0
    a, e, I, _L, wbar, Om = (el0[i] + rate[i] * T for i in range(6))
    omega = np.radians(wbar - Om)
    cO, sO, cI, sI = (np.cos(np.radians(Om)), np.sin(np.radians(Om)),
                      np.cos(np.radians(I)), np.sin(np.radians(I)))
    out = []
    for E in (0.0, np.pi):
        xp, yp = a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E)
        co, so = np.cos(omega), np.sin(omega)
        out.append(np.array([(co * cO - so * sO * cI) * xp + (-so * cO - co * sO * cI) * yp,
                             (co * sO + so * cO * cI) * xp + (-so * sO + co * cO * cI) * yp,
                             (so * sI) * xp + (co * sI) * yp]) * AU)
    return out[0], out[1]


def ephemeris_spec() -> dict:
    """Datos para que el navegador calcule posiciones (mismas fórmulas que ``positions``)."""
    mm, me = _PHYS["luna"][1], _PHYS["tierra"][1]
    return {"elements": {k: [v[0], v[1]] for k, v in _EL.items()},
            "moon": {"dist": _MOON_DIST, "inc": float(_MOON_INC), "ratio": mm / (me + mm)}}
