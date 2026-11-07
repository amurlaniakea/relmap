# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
import numpy as np
import pytest

from relmap import (Body, integrate_null, load_scene, monte_carlo, parse_scene,
                    potential, time_dilation, trace_ray)
from relmap.scene import Ray

M_SUN, R_SUN = 1.98847e30, 6.957e8


def sun():
    return Body("Sol", M_SUN, np.zeros(3), radius=R_SUN)


def point(m_geo, pos, name="p"):
    from relmap.units import G, C
    return Body(name, m_geo * C**2 / G, np.array(pos, float))


def test_deflexion_sol_175():
    r = trace_ray([sun()], [-1e5 * R_SUN, R_SUN, 0], [1, 0, 0], 2e5 * R_SUN)
    assert r.deflection_arcsec == pytest.approx(1.7505, abs=0.01)


def test_coincide_con_schwarzschild_fase1():
    M, b, r0 = 1.0, 1e4, 1e7
    _, d1 = integrate_null(M, r0=r0, b=b)
    r = trace_ray([point(M, [0, 0, 0])], [-r0, b, 0], [1, 0, 0], 2 * r0)
    assert r.deflection_rad == pytest.approx(d1, rel=1e-3)


def test_simetria_dos_masas_cancela():
    D = 1e4
    bodies = [point(1.0, [0, D, 0], "a"), point(1.0, [0, -D, 0], "b")]
    r = trace_ray(bodies, [-1e7, 0, 0], [1, 0, 0], 2e7)
    assert r.deflection_rad < 1e-9           # una sola masa daría ~4e-4


def test_superposicion_lejana():
    M, b, D = 1.0, 1e4, 1e9
    one = trace_ray([point(M, [0, 0, 0])], [-1e7, b, 0], [1, 0, 0], 2e7).deflection_rad
    two = trace_ray([point(M, [0, 0, 0]), point(M, [0, 0, D], "lejana")],
                    [-1e7, b, 0], [1, 0, 0], 2e7).deflection_rad
    assert two == pytest.approx(one, rel=1e-3)


def test_retraso_shapiro_analitico():
    M, b, a = 1.0, 1e4, 1e7
    r = trace_ray([point(M, [0, 0, 0])], [-a, b, 0], [1, 0, 0], 2 * a)
    assert r.delay_m == pytest.approx(4 * M * np.arcsinh(a / b), rel=2e-3)


def test_dilatacion_temporal_tierra():
    earth = Body("Tierra", 5.9722e24, np.zeros(3), radius=6.371e6)
    f = time_dilation([6.371e6, 0, 0], [earth])
    assert 1 - f == pytest.approx(6.95e-10, rel=1e-2)


def test_potencial_continuo_en_superficie():
    b = sun()
    out = potential([R_SUN * 1.0000001, 0, 0], [b])
    inn = potential([R_SUN * 0.9999999, 0, 0], [b])
    assert out == pytest.approx(inn, rel=1e-6)


def test_impacto_detectado():
    r = trace_ray([sun()], [-1e5 * R_SUN, 0.5 * R_SUN, 0], [1, 0, 0], 2e5 * R_SUN)
    assert r.hit == "Sol" and np.isnan(r.deflection_rad)


def test_montecarlo_propaga_incertidumbre_de_masa():
    b = Body("p", 1.0e30, np.zeros(3), mass_sigma_rel=0.10)
    from relmap.units import G, C
    ray = Ray("r", np.array([-1e12, 1e9, 0]), np.array([1.0, 0, 0]), 2e12)
    s = monte_carlo([b], ray, n=80, seed=3)
    d = s.deflection_arcsec
    assert d["std"] / d["mean"] == pytest.approx(0.10, rel=0.4)   # deflexión ∝ masa


def test_montecarlo_sin_incertidumbre_es_determinista():
    b = Body("p", 1.0e30, np.zeros(3))
    ray = Ray("r", np.array([-1e12, 1e9, 0]), np.array([1.0, 0, 0]), 2e12)
    s = monte_carlo([b], ray, n=5, seed=1)
    assert s.deflection_arcsec["std"] == pytest.approx(0.0, abs=1e-12)


def test_carga_yaml(tmp_path):
    f = tmp_path / "e.yaml"
    f.write_text("unit: au\nbodies:\n  - {name: Sol, mass: 1, mass_unit: msun, position: [0,0,0]}\n"
                 "rays:\n  - {name: r, origin: [-1,0.1,0], direction: [2,0,0], length: 2}\n",
                 encoding="utf-8")
    sc = load_scene(f)
    assert sc.bodies[0].mass_kg == pytest.approx(M_SUN)
    assert sc.rays[0].origin[0] == pytest.approx(-1.495978707e11)
    assert np.allclose(sc.rays[0].direction, [1, 0, 0])


@pytest.mark.parametrize("bad", [
    {"unit": "parsecs", "bodies": []},
    {"bodies": [{"mass": -1, "position": [0, 0, 0]}]},
    {"bodies": [{"mass": 1, "mass_unit": "stones", "position": [0, 0, 0]}]},
    {"bodies": [{"mass": 1, "position": [0, 0]}]},
    {"rays": [{"origin": [0, 0, 0], "direction": [0, 0, 0], "length": 1}]},
])
def test_escena_invalida(bad):
    with pytest.raises(ValueError):
        parse_scene(bad)
