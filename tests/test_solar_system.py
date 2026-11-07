# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
from datetime import datetime, timezone

import numpy as np
import pytest

from relmap import parse_scene, scene_to_dict, solar_system_bodies, trace_ray
from relmap.scene import Scene
from relmap.solar_system import (
    _EL,
    _PHYS,
    AU,
    _helio,
    default_rays,
    julian_date,
    positions,
    solve_kepler,
)

UTC = timezone.utc


def dt(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def test_julian_date_j2000():
    assert julian_date(dt("2000-01-01T12:00")) == pytest.approx(2451545.0)


@pytest.mark.parametrize("e", [0.0, 0.0068, 0.2056, 0.5])
@pytest.mark.parametrize("M", [-170.0, -10.0, 0.0, 33.3, 179.0])
def test_kepler_ida_y_vuelta(M, e):
    E = solve_kepler(M, e)
    assert E - np.degrees(e) * np.sin(np.radians(E)) == pytest.approx(M, abs=1e-8)


def test_tierra_perihelio_enero_2026():
    r = np.linalg.norm(positions(dt("2026-01-03"))["tierra"]) / AU
    assert r == pytest.approx(0.9833, abs=1e-3)


def test_tierra_afelio_julio_2026():
    r = np.linalg.norm(positions(dt("2026-07-06"))["tierra"]) / AU
    assert r == pytest.approx(1.0167, abs=1.5e-3)


@pytest.mark.parametrize("key", ["mercurio", "venus", "marte", "jupiter", "saturno", "urano", "neptuno"])
@pytest.mark.parametrize("date", ["1900-06-01", "2000-01-01", "2026-11-05", "2049-03-03"])
def test_distancia_dentro_de_apsides(key, date):
    from relmap.solar_system import julian_date as jd
    a, e = _EL[key][0][0], _EL[key][0][1]
    r = np.linalg.norm(_helio(key, jd(dt(date)))) / AU
    assert a * (1 - e) * 0.995 <= r <= a * (1 + e) * 1.005


def test_periodo_de_jupiter():
    p0 = _helio("jupiter", julian_date(dt("2020-01-01")))
    p1 = _helio("jupiter", julian_date(dt("2020-01-01")) + 11.862615 * 365.25)
    ang = np.degrees(np.arccos(np.dot(p0, p1) / (np.linalg.norm(p0) * np.linalg.norm(p1))))
    assert ang < 1.0


def test_luna_y_baricentro_tierra_luna():
    p = positions(dt("2026-11-05"))
    assert np.linalg.norm(p["luna"] - p["tierra"]) == pytest.approx(384.4e6, rel=1e-9)
    mm, me = _PHYS["luna"][1], _PHYS["tierra"][1]
    bary = (me * p["tierra"] + mm * p["luna"]) / (me + mm)
    emb = _helio("embary", julian_date(dt("2026-11-05")))
    assert np.linalg.norm(bary - emb) < 1.0           # metros: coincide con el EMB de JPL
    assert np.linalg.norm(p["tierra"] - emb) == pytest.approx(4.67e6, rel=0.01)


def test_cuerpos_completos_y_sin_solapes():
    bodies = solar_system_bodies(dt("2026-11-05"))
    assert [b.name for b in bodies][:2] == ["Sol", "Mercurio"] and len(bodies) == 11
    for i, a in enumerate(bodies):
        for b in bodies[i + 1:]:
            assert np.linalg.norm(a.position - b.position) > a.radius + b.radius


def test_include_acepta_nombres_sin_tilde():
    assert [b.name for b in solar_system_bodies(dt("2026-11-05"), include=["sol", "JUPITER"])] == ["Sol", "Júpiter"]
    with pytest.raises(ValueError):
        solar_system_bodies(dt("2026-11-05"), include=["Ceres"])


def test_rasante_sol_deflexion_875():
    bodies = solar_system_bodies(dt("2026-11-05"))
    ray = next(r for r in default_rays(bodies) if "Sol" in r.name)
    res = trace_ray(bodies, ray.origin, ray.direction, ray.length)
    assert res.deflection_arcsec == pytest.approx(1.7505 / 2, rel=0.03)    # 4M/(2R)


def test_rasante_jupiter_solo_coincide_con_formula():
    bodies = solar_system_bodies(dt("2026-11-05"), include=["Júpiter"])
    b = bodies[0]
    ray = default_rays(bodies, b_factor=3.0)[0]
    res = trace_ray(bodies, ray.origin, ray.direction, ray.length)
    ref = np.degrees(4 * b.mass_geo / (3.0 * b.radius)) * 3600
    assert res.deflection_arcsec == pytest.approx(ref, rel=2e-3)


def test_exportar_y_recargar_escena():
    bodies = solar_system_bodies(dt("2026-11-05"))
    sc = Scene(bodies=bodies, rays=default_rays(bodies))
    back = parse_scene(scene_to_dict(sc))
    assert len(back.bodies) == len(bodies) and len(back.rays) == len(sc.rays)
    for x, y in zip(sc.bodies, back.bodies):
        assert np.allclose(x.position, y.position, rtol=1e-12, atol=1.0)
        assert y.mass_kg == pytest.approx(x.mass_kg, rel=1e-12)
