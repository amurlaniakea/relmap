# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""relmap-pdf: genera la guía pedagógica (PDF) para estudiantes y profesores."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="relmap-pdf", description="Guía pedagógica en PDF")
    ap.add_argument("--nivel", choices=["estudiante", "profesor", "ambos"], default="ambos")
    ap.add_argument("--date", default=None, help="fecha ISO del Sistema Solar de ejemplo (por defecto: hoy UTC)")
    ap.add_argument("--out", default=None, help="archivo de salida (con --nivel ambos se añade el sufijo)")
    a = ap.parse_args(argv)
    try:
        from .pedagogy import build_pdf
    except ImportError as e:
        print(f"Faltan dependencias del PDF ({e}). Instala:  pip install \"relmap[pdf]\"  (reportlab y matplotlib)", file=sys.stderr)
        return 2
    when = datetime.fromisoformat(a.date).replace(tzinfo=timezone.utc) if a.date else datetime.now(timezone.utc)
    niveles = ["estudiante", "profesor"] if a.nivel == "ambos" else [a.nivel]
    for n in niveles:
        out = (a.out or "guia_relmap.pdf")
        if a.nivel == "ambos" or a.out is None:
            stem = out[:-4] if out.lower().endswith(".pdf") else out
            out = f"{stem}_{n}.pdf" if a.out is None or a.nivel == "ambos" else out
        print(f"Generando {out} ({n})…")
        build_pdf(out, nivel=n, when=when)
        print(f"  listo: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
