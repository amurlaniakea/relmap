# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""Línea de comandos:  relmap escena.yaml [--mc N] [--out mapa.html]"""
from __future__ import annotations

import argparse
import sys

from .scene import load_scene
from .scene_map import build_scene_map
from .uncertainty import monte_carlo
from .weakfield import trace_ray, time_dilation


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="relmap", description="Navegación relativista en campo débil")
    ap.add_argument("scene", help="escena YAML/JSON")
    ap.add_argument("--mc", type=int, default=0, help="muestras Monte Carlo (0 = sin incertidumbre)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="mapa.html")
    ap.add_argument("--exaggerate", default="auto", help="factor de exageración de desviaciones o 'auto'")
    ap.add_argument("--offline", action="store_true", help="incrusta plotly.js (funciona sin internet)")
    a = ap.parse_args(argv)

    scene = load_scene(a.scene)
    if not scene.bodies:
        print("La escena no tiene cuerpos.", file=sys.stderr)
        return 2
    print(f"{len(scene.bodies)} cuerpos, {len(scene.rays)} rayos\n")
    results = []
    for ray in scene.rays:
        r = trace_ray(scene.bodies, ray.origin, ray.direction, ray.length, name=ray.name)
        results.append(r)
        if r.hit:
            print(f"[{ray.name}] IMPACTA contra {r.hit}")
            continue
        print(f"[{ray.name}] deflexión = {r.deflection_arcsec:.6g}\"   retraso = {r.delay_s*1e6:.6g} µs")
        if a.mc > 0:
            s = monte_carlo(scene.bodies, ray, n=a.mc, seed=a.seed)
            d, t = s.deflection_arcsec, s.delay_s
            print(f"    MC n={s.n}: deflexión {d['p50']:.6g}\" (90%: {d['p5']:.6g}–{d['p95']:.6g}, σ={d['std']:.3g})"
                  f"; retraso {t['p50']*1e6:.6g} µs (90%: {t['p5']*1e6:.6g}–{t['p95']*1e6:.6g})"
                  + (f"; {s.hits} impactos" if s.hits else ""))
    print()
    for b in scene.bodies:
        if b.radius > 0:
            surf = b.position + [b.radius, 0.0, 0.0]
            f = time_dilation(surf, [b])
            print(f"[{b.name}] reloj en su superficie va más lento en {1 - f:.3e} (fracción)")
        else:
            print(f"[{b.name}] cuerpo puntual (sin radio): sin dilatación en superficie")
    build_scene_map(scene, results, exaggerate=a.exaggerate, out_html=a.out, offline=a.offline)
    print(f"\nGenerado {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
