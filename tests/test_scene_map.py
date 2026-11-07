# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
from datetime import datetime, timezone

import numpy as np
import pytest

from relmap import default_rays, solar_system_bodies, trace_ray
from relmap.scene import Body, Scene
from relmap.scene_map import SUN_YELLOW, body_colors, build_scene_map, default_views
from relmap.solar_system import (
    AU,
    BELT_INNER_AU,
    BELT_OUTER_AU,
    apsides,
    asteroid_belt,
    ephemeris_spec,
    julian_date,
    orbit_elements,
    orbit_path,
)

WHEN = datetime(2026, 11, 5, tzinfo=timezone.utc)
BELT = {"inner_m": BELT_INNER_AU * AU, "outer_m": BELT_OUTER_AU * AU}


@pytest.fixture(scope="module")
def solar():
    bodies = solar_system_bodies(WHEN)
    rays = default_rays(bodies)
    res = [trace_ray(bodies, r.origin, r.direction, r.length, name=r.name) for r in rays]
    orbits = [(b.name, p) for b in bodies if (p := orbit_path(b.name, WHEN)) is not None]
    return Scene(bodies=bodies, rays=rays), res, orbits


def test_amarillo_solo_para_el_sol(solar):
    cols = body_colors(solar[0].bodies)
    assert [n for n, c in cols.items() if c.lower() == SUN_YELLOW.lower()] == ["Sol"]
    assert len(set(cols.values())) == len(cols)


def test_fallback_sin_amarillo():
    bs = [Body(f"x{i}", 1e24, np.array([i, 0, 0.0])) for i in range(12)]
    assert SUN_YELLOW.lower() not in [c.lower() for c in body_colors(bs).values()]


def test_tres_vistas_principales_y_una_por_cuerpo(solar):
    scene, res, orbits = solar
    views = default_views(scene.bodies, res, belt=BELT, orbits=orbits)
    names = [v["name"] for v in views]
    assert names[:3] == ["1 · Todo el sistema", "2 · Sistema solar interior", "3 · Sistema solar exterior"]
    assert all(f"Centrar en {b.name}" in names for b in scene.bodies)
    inner, outer = views[1]["include"], views[2]["include"]
    assert {"Sol", "Mercurio", "Venus", "Tierra", "Luna", "Marte"} == set(inner)
    assert {"Sol", "Júpiter", "Saturno", "Urano", "Neptuno", "Plutón"} == set(outer)
    assert views[1]["h"] < 4 * AU and views[2]["h"] > 40 * AU          # zoom real a cada zona


def test_sin_cinturon_solo_vista_general(solar):
    scene, res, orbits = solar
    views = default_views(scene.bodies, res, belt=None, orbits=orbits)
    assert [v["name"] for v in views[:2]] == ["1 · Todo el sistema", "Centrar en Sol"]


def test_cfg_animacion_cinturon_y_apsides(solar):
    scene, res, orbits = solar
    aps = [(b.name, *apsides(b.name, WHEN)) for b in scene.bodies
           if orbit_elements(b.name, WHEN) and orbit_elements(b.name, WHEN)[1] >= 0.05]
    belt = {**BELT, "points": asteroid_belt(500)}
    fig = build_scene_map(scene, res, orbits=orbits, ephemeris=ephemeris_spec(), jd0=julian_date(WHEN),
                          belt=belt, apsides=aps)
    cfg = fig._relmap_cfg
    assert cfg["dynamic"] and cfg["earthName"] == "Tierra" and cfg["moonName"] == "Luna"
    assert len(cfg["belt"]) == 500 * 3
    assert {a["name"] for a in cfg["apsides"]} == {"Mercurio", "Marte", "Saturno", "Plutón"}
    assert cfg["orbits"]["Luna"]["pts"] == []                       # la Luna se calcula en el navegador
    assert "pluton" in cfg["eph"]["elements"]
    roles = [t.meta["role"] for t in fig.data]
    assert roles.count("body") == 11 and roles.count("grid") == 1 and "belt" in roles


def test_html_modo_oscuro_y_controles(solar, tmp_path):
    scene, res, orbits = solar
    out = tmp_path / "m.html"
    fig = build_scene_map(scene, res, orbits=orbits, out_html=str(out), ephemeris=ephemeris_spec(),
                          jd0=julian_date(WHEN), belt={**BELT, "points": asteroid_belt(200)})
    assert fig.layout.paper_bgcolor == "#0b1020"
    html = out.read_text(encoding="utf-8")
    for needle in ("Modo oscuro", "inp.checked = true", "Movimiento de traslación", "rm-panel", "optgroup"):
        assert needle in html


def test_escena_estatica_sin_efemerides_no_se_anima():
    bs = [Body("A", 1e30, np.zeros(3), radius=1e6), Body("B", 1e29, np.array([1e9, 0, 0.0]), radius=1e5)]
    fig = build_scene_map(Scene(bodies=bs), [])
    assert fig._relmap_cfg["dynamic"] is False


# ---- catálogo: Plutón y cinturón ----
def test_pluton_en_el_catalogo():
    p = next(b for b in solar_system_bodies(WHEN) if b.name == "Plutón")
    assert 30 < np.linalg.norm(p.position) / AU < 50
    assert p.radius == pytest.approx(1188.3e3)
    a, e, q, Q = orbit_elements("Plutón", WHEN)
    assert (a, e) == pytest.approx((39.48, 0.2488), rel=1e-3) and q < 30.2 < Q    # q < a(Neptuno)


def test_cinturon_dentro_de_limites_y_con_huecos():
    pts = asteroid_belt(4000, seed=3)
    a_proxy = np.linalg.norm(pts, axis=1) / AU
    assert 1.2 < a_proxy.min() and a_proxy.max() < 4.6
    assert np.median(a_proxy) == pytest.approx(2.7, abs=0.25)
    assert np.allclose(asteroid_belt(50, seed=9), asteroid_belt(50, seed=9))      # reproducible


def test_apsides_coinciden_con_elementos():
    peri, afe = apsides("Marte", WHEN)
    _a, _e, q, Q = orbit_elements("Marte", WHEN)
    assert np.linalg.norm(peri) / AU == pytest.approx(q, rel=1e-9)
    assert np.linalg.norm(afe) / AU == pytest.approx(Q, rel=1e-9)


def test_orbita_pluton_entre_perihelio_y_afelio():
    p = np.linalg.norm(orbit_path("Plutón", WHEN), axis=1) / AU
    q, Q = orbit_elements("Plutón", WHEN)[2:]
    assert p.min() == pytest.approx(q, rel=2e-3) and p.max() == pytest.approx(Q, rel=2e-3)
