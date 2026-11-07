# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
import numpy as np
import pytest

from relmap import integrate_null, integrate_timelike, schwarzschild_mass_length

M_SUN = 1.98847e30
R_SUN = 6.957e8


def test_masa_geometrica_sol():
    assert schwarzschild_mass_length(M_SUN) == pytest.approx(1476.6, rel=1e-3)


def test_deflexion_luz_sol_175_arcsec():
    M = schwarzschild_mass_length(M_SUN)
    _, defl = integrate_null(M, r0=1e5 * R_SUN, b=R_SUN)
    arcsec = np.degrees(defl) * 3600
    assert arcsec == pytest.approx(1.7505, abs=0.01)  # 4M/b = 1.75"


def test_deflexion_campo_debil_analitica():
    M, b = 1.0, 1e4
    _, defl = integrate_null(M, r0=1e7, b=b)
    esperado = 4 * M / b + 15 * np.pi / 4 * (M / b) ** 2
    assert defl == pytest.approx(esperado, rel=1e-3)


def test_fotón_capturado():
    _, defl = integrate_null(1.0, r0=1e4, b=2.0)
    assert np.isnan(defl)


def test_precesion_periastro_6piM_sobre_p():
    M, a, e = 1.0, 1000.0, 0.1
    _, phis = integrate_timelike(M, a * (1 - e), a * (1 + e), n_orbits=4)
    assert len(phis) >= 3
    medido = np.diff(phis) - 2 * np.pi
    esperado = 6 * np.pi * M / (a * (1 - e**2))
    assert np.allclose(medido, esperado, rtol=2e-2)
