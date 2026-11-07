# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""relmap-solar: el Sistema Solar en una fecha, con rayos rasantes y mapa 3D."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np

from .scene import Scene, load_scene
from .scene_map import build_scene_map
from .solar_system import (
    AU,
    BELT_INNER_AU,
    BELT_OUTER_AU,
    apsides,
    asteroid_belt,
    default_rays,
    ephemeris_spec,
    julian_date,
    orbit_elements,
    orbit_path,
    scene_to_dict,
    solar_system_bodies,
)
from .uncertainty import monte_carlo
from .weakfield import time_dilation, trace_ray


def _date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="relmap-solar", description="Sistema Solar en campo débil")
    ap.add_argument("--date", type=_date, default=None, help="fecha ISO, p. ej. 2026-11-05 (por defecto: ahora UTC)")
    ap.add_argument("--include", nargs="*", help="solo estos cuerpos (p. ej. Sol Tierra Luna)")
    ap.add_argument("--b-factor", type=float, default=2.0, help="parámetro de impacto de los rayos, en radios del cuerpo")
    ap.add_argument("--rays", help="YAML/JSON con tus propios rayos (marco eclíptico J2000 heliocéntrico)")
    ap.add_argument("--mc", type=int, default=0, help="muestras Monte Carlo por rayo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="sistema_solar.html")
    ap.add_argument("--dump", help="guarda la escena generada en este YAML")
    ap.add_argument("--exaggerate", default="auto")
    ap.add_argument("--no-belt", action="store_true", help="no dibujar el cinturón de asteroides")
    ap.add_argument("--offline", action="store_true", help="incrusta plotly.js en el HTML (funciona sin internet, ~4 MB)")
    a = ap.parse_args(argv)

    when = a.date or datetime.now(timezone.utc)
    bodies = solar_system_bodies(when, include=a.include)
    rays = load_scene(a.rays).rays if a.rays else default_rays(bodies, b_factor=a.b_factor)
    scene = Scene(bodies=bodies, rays=rays)
    print(f"Sistema Solar a {when:%Y-%m-%d %H:%M} UTC — {len(bodies)} cuerpos, {len(rays)} rayos")
    print("Marco: eclíptico J2000 heliocéntrico (JPL aproximado; ver README para errores)\n")

    if a.dump:
        import yaml
        with open(a.dump, "w", encoding="utf-8") as f:
            yaml.safe_dump(scene_to_dict(scene), f, sort_keys=False, allow_unicode=True)
        print(f"Escena guardada en {a.dump}\n")

    results = []
    print("«total» = todo el sistema actuando sobre el rayo; «aporte» = lo que añade el propio cuerpo")
    print("(total − mismo rayo sin ese cuerpo). Para cuerpos pequeños el total lo domina el Sol.\n")
    print(f"{'rayo':34s} {'total (″)':>11s} {'aporte (″)':>11s} {'4GM/bc² (″)':>12s} {'retraso (µs)':>13s}")

    def final_dir(res):
        p = res.sol.y[3:6, -1]
        return p / np.linalg.norm(p)

    for ray in rays:
        r = trace_ray(bodies, ray.origin, ray.direction, ray.length, name=ray.name)
        results.append(r)
        if r.hit:
            print(f"{ray.name:34s} IMPACTA contra {r.hit}")
            continue
        own, ref = "", ""
        for b in bodies:
            if ray.name.startswith(f"rasante {b.name} "):
                others = [x for x in bodies if x is not b]
                # aportes de ~1e-11 rad: se usa una tolerancia más fina que la normal
                r_hi = trace_ray(bodies, ray.origin, ray.direction, ray.length, rtol=3e-14)
                r2 = trace_ray(others, ray.origin, ray.direction, ray.length, rtol=3e-14)
                d = np.degrees(np.arctan2(np.linalg.norm(np.cross(final_dir(r_hi), final_dir(r2))),
                                          float(np.dot(final_dir(r_hi), final_dir(r2))))) * 3600
                own = f"{d:11.4g}"
                ref = f"{np.degrees(4 * b.mass_geo / (a.b_factor * b.radius)) * 3600:12.4g}"
        print(f"{ray.name:34s} {r.deflection_arcsec:11.4g} {own:>11s} {ref:>12s} {r.delay_s * 1e6:13.5g}")
        if a.mc > 0:
            s_ = monte_carlo(bodies, ray, n=a.mc, seed=a.seed)
            d = s_.deflection_arcsec
            print(f"{'':34s} MC n={s_.n}: mediana {d['p50']:.4g}″, 90%: {d['p5']:.4g}–{d['p95']:.4g}″"
                  + (f", {s_.hits} impactos" if s_.hits else ""))

    print("\nDilatación temporal en la superficie (1 − dτ/dt):")
    for b in bodies:
        if b.radius > 0:
            f = time_dilation(b.position + [b.radius, 0.0, 0.0], bodies)
            print(f"  {b.name:10s} {1 - f:.4e}")
    orbits = [(b.name, p) for b in bodies if (p := orbit_path(b.name, when)) is not None]
    info, aps = {}, []
    print("\nÓrbitas (elipses con el Sol en un foco): semieje a, excentricidad e, perihelio q, afelio Q")
    for b in bodies:
        oe = orbit_elements(b.name, when)
        if oe is None:
            continue
        a_, e_, q_, Q_ = oe
        info[b.name] = f"a={a_:.4g} UA, e={e_:.4f}, q={q_:.4g} UA, Q={Q_:.4g} UA"
        print(f"  {b.name:10s} a={a_:8.4f}  e={e_:.4f}  q={q_:8.4f}  Q={Q_:8.4f} UA")
        if e_ >= 0.05:
            ap = apsides(b.name, when)
            aps.append((b.name, ap[0], ap[1]))
    belt = None
    if not a.no_belt and a.include is None:
        belt = {"inner_m": BELT_INNER_AU * AU, "outer_m": BELT_OUTER_AU * AU, "points": asteroid_belt()}
        print(f"\nCinturón principal de asteroides: {BELT_INNER_AU}–{BELT_OUTER_AU} UA "
              "(nube ilustrativa; separa el sistema interior del exterior)")
    build_scene_map(scene, results, exaggerate=a.exaggerate, out_html=a.out, orbits=orbits,
                    offline=a.offline, title=f"Sistema Solar {when:%Y-%m-%d}", ephemeris=ephemeris_spec(),
                    jd0=julian_date(when), belt=belt, apsides=aps, orbit_info=info)
    print(f"\nGenerado {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
