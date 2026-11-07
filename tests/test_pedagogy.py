# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
import pytest

pytest.importorskip("reportlab")
pytest.importorskip("matplotlib")

from datetime import datetime, timezone


def test_pdf_estudiante_se_genera(tmp_path):
    from relmap.pedagogy import build_pdf
    out = tmp_path / "g.pdf"
    build_pdf(str(out), nivel="estudiante", when=datetime(2026, 11, 5, tzinfo=timezone.utc), log=lambda m: None)
    data = out.read_bytes()
    assert data[:5] == b"%PDF-" and len(data) > 200_000


def test_datos_del_pdf_coinciden_con_la_fisica():
    from relmap.pedagogy import compute
    D = compute(datetime(2026, 11, 5, tzinfo=timezone.utc), log=lambda m: None)
    assert D.sun_limb["num"] == pytest.approx(D.sun_limb["formula"], rel=0.01)
    assert D.mercury["num"] == pytest.approx(42.98, abs=0.05) and D.mercury["ana"] == pytest.approx(42.98, abs=0.05)
    assert [r["name"] for r in D.table][0] == "Sol" and len(D.table) == 11
    for r in D.table:                                   # el «aporte» propio coincide con 4GM/(bc²)
        assert r["own"] == pytest.approx(r["formula"], rel=0.03)
