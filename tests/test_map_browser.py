# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""Prueba de integración en un navegador real (se omite si no hay Playwright/Chromium)."""
from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("playwright.sync_api")

from relmap import default_rays, solar_system_bodies, trace_ray
from relmap.scene import Scene
from relmap.scene_map import build_scene_map
from relmap.solar_system import (
    AU,
    BELT_INNER_AU,
    BELT_OUTER_AU,
    asteroid_belt,
    ephemeris_spec,
    julian_date,
    orbit_path,
    positions,
)

NAMES = {"Sol": "sol", "Mercurio": "mercurio", "Venus": "venus", "Tierra": "tierra", "Luna": "luna", "Marte": "marte",
         "Júpiter": "jupiter", "Saturno": "saturno", "Urano": "urano", "Neptuno": "neptuno", "Plutón": "pluton"}


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    from playwright.sync_api import sync_playwright
    when = datetime(2026, 11, 5, tzinfo=timezone.utc)
    bodies = solar_system_bodies(when)
    rays = default_rays(bodies)
    res = [trace_ray(bodies, r.origin, r.direction, r.length, name=r.name) for r in rays]
    orbits = [(b.name, p) for b in bodies if (p := orbit_path(b.name, when)) is not None]
    html = tmp_path_factory.mktemp("m") / "m.html"
    build_scene_map(Scene(bodies=bodies, rays=rays), res, orbits=orbits, out_html=str(html), offline=True,
                    ephemeris=ephemeris_spec(), jd0=julian_date(when),
                    belt={"inner_m": BELT_INNER_AU * AU, "outer_m": BELT_OUTER_AU * AU, "points": asteroid_belt(300)})
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        except (OSError, RuntimeError) as e:                                    # pragma: no cover
            pytest.skip(f"sin Chromium: {e}")
        pg = b.new_page(viewport={"width": 1200, "height": 800})
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto("file://" + str(html))
        pg.wait_for_selector("#rm-panel", timeout=60000)
        pg.wait_for_timeout(3000)
        yield pg, errs
        b.close()


def test_sin_errores_de_javascript(page):
    _pg, errs = page
    assert errs == []


def test_efemerides_js_coinciden_con_python(page):
    pg, _ = page
    for d in ("1850-03-17", "2000-01-01", "2026-11-05", "2049-07-01"):
        t = datetime.fromisoformat(d).replace(tzinfo=timezone.utc)
        py, js = positions(t), pg.evaluate("(jd)=>window.__rm.positions(jd)", julian_date(t))
        for n, k in NAMES.items():
            assert np.linalg.norm(np.array(js[n]) - py[k]) < 1.0       # menos de 1 m


def test_las_tres_vistas_y_la_animacion(page):
    pg, errs = page
    opts = pg.eval_on_selector_all("#rm-panel select option", "e=>e.map(o=>o.textContent)")
    assert opts[:3] == ["1 · Todo el sistema", "2 · Sistema solar interior", "3 · Sistema solar exterior"]
    pg.select_option("#rm-panel select", "1")
    pg.wait_for_timeout(1500)
    d0 = pg.inner_text("#rm-date")
    pg.evaluate("()=>{window.__rm.state.jd += 365.25*3; window.__rm.render(false);}")
    pg.wait_for_timeout(800)
    pg.evaluate("()=>window.__rm.render(false)")
    assert errs == [] and d0 == "2026-11-05"
