# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""Guía pedagógica en PDF: cómo se resuelven las ecuaciones y cómo usar relmap en clase.

Todos los números (tablas, ejemplos, soluciones) se calculan en el momento con el propio código de
relmap, así el documento no puede contradecir al programa. Requiere ``pip install relmap[pdf]``
(reportlab + matplotlib). Dos niveles: ``estudiante`` y ``profesor`` (añade derivaciones,
soluciones, errores frecuentes y sugerencias de aula).
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from . import __version__
from .geodesics import integrate_null, integrate_timelike
from .solar_system import (
    _EL,
    AU,
    BELT_INNER_AU,
    BELT_OUTER_AU,
    default_rays,
    julian_date,
    orbit_elements,
    orbit_path,
    solar_system_bodies,
    solve_kepler,
)
from .uncertainty import monte_carlo
from .units import C, schwarzschild_mass_length
from .weakfield import time_dilation, trace_ray

ARCSEC = 180.0 / np.pi * 3600.0
M_SUN, R_SUN = 1.98847e30, 6.957e8


# ------------------------------------------------------------------ datos calculados con relmap
@dataclass
class Data:
    when: datetime
    gm: list
    sun_limb: dict
    schw: dict
    mercury: dict
    table: list
    dilation: list
    mc: dict
    kepler: dict
    ellipse: list
    answers: dict


def compute(when: datetime, log=print) -> Data:
    log("· cantidades geométricas…")
    gm = [(n, m, schwarzschild_mass_length(m)) for n, m in
          (("Sol", M_SUN), ("Júpiter", 1.89813e27), ("Tierra", 5.97217e24), ("Luna", 7.346e22))]
    Msun = schwarzschild_mass_length(M_SUN)

    log("· deflexión de la luz rozando el Sol (campo débil)…")
    b_sun = 1.001 * R_SUN                      # un pelo por fuera del limbo, para no rozar la superficie
    res = trace_ray([_sun_body()], [-1e4 * R_SUN, b_sun, 0], [1, 0, 0], 2e4 * R_SUN)
    sun_limb = {"num": res.deflection_arcsec, "formula": 4 * Msun / b_sun * ARCSEC,
                "newton": 2 * Msun / b_sun * ARCSEC}

    log("· Schwarzschild exacto frente a campo débil…")
    M, b, r0 = 1.0, 1e4, 1e7
    _, d_exact = integrate_null(M, r0=r0, b=b)
    schw = {"b": b, "exact": d_exact, "first": 4 * M / b, "second": 4 * M / b + 15 * np.pi / 4 * (M / b) ** 2}

    log("· precesión del perihelio de Mercurio…")
    a, e = _EL["mercurio"][0][0] * AU, _EL["mercurio"][0][1]
    _, phis = integrate_timelike(Msun, a * (1 - e), a * (1 + e), n_orbits=3)
    per_orb = float(np.mean(np.diff(phis) - 2 * np.pi))
    ana = 6 * np.pi * Msun / (a * (1 - e ** 2))
    orbits_cy = 36525.0 / 87.969
    mercury = {"num": per_orb * orbits_cy * ARCSEC, "ana": ana * orbits_cy * ARCSEC, "orb": per_orb}

    log("· Sistema Solar en la fecha elegida (10 rayos rasantes)…")
    bodies = solar_system_bodies(when)
    table = []
    for ray in default_rays(bodies):
        r = trace_ray(bodies, ray.origin, ray.direction, ray.length, name=ray.name)
        nm = ray.name.split()[1]
        bd = next(x for x in bodies if x.name == nm)
        others = [x for x in bodies if x is not bd]
        # aportes de ~1e-11 rad: hace falta más precisión que la de uso normal (rtol 1e-12)
        r_hi = trace_ray(bodies, ray.origin, ray.direction, ray.length, rtol=3e-14)
        r2 = trace_ray(others, ray.origin, ray.direction, ray.length, rtol=3e-14)
        f1, f2 = (_fdir(r_hi), _fdir(r2))
        own = float(np.arctan2(np.linalg.norm(np.cross(f1, f2)), np.dot(f1, f2)) * ARCSEC)
        table.append({"name": nm, "total": r.deflection_arcsec, "own": own,
                      "formula": 4 * bd.mass_geo / (2 * bd.radius) * ARCSEC, "delay_us": r.delay_s * 1e6})
    dilation = [(x.name, 1 - time_dilation(x.position + [x.radius, 0, 0], [x])) for x in bodies if x.radius > 0]

    log("· incertidumbre (Monte Carlo)…")
    [bd for bd in bodies if bd.name == "Sol"]
    mc = {}
    from dataclasses import replace

    from .scene import Ray
    sb = [replace(_sun_body(), mass_sigma_rel=0.01)]
    ray = Ray("r", np.array([-2e3 * R_SUN, 1.5 * R_SUN, 0.0]), np.array([1.0, 0, 0]), 4e3 * R_SUN)
    s = monte_carlo(sb, ray, n=40, seed=1)
    mc["sol"] = (s.deflection_arcsec["mean"], s.deflection_arcsec["std"])
    jup = [x for x in solar_system_bodies(when, include=["Júpiter"])]
    jb = jup[0]
    ray_j = default_rays(jup, b_factor=2.0)[0]
    out = []
    for sig_km in (0.0, 5.0e3, 3.0e5):
        j2 = [replace(jb, position_sigma=sig_km * 1e3)]
        sj = monte_carlo(j2, ray_j, n=40, seed=2)
        out.append((sig_km, sj.deflection_arcsec["mean"], sj.deflection_arcsec["std"]))
    mc["jup"] = out

    log("· ecuación de Kepler y elipses…")
    el0, rate = _EL["marte"]
    T = (julian_date(when) - 2451545.0) / 36525.0
    a_, e_, _, L_, wb_, __ = (el0[i] + rate[i] * T for i in range(6))
    Mdeg = (L_ - wb_ + 180.0) % 360.0 - 180.0
    estar, Eit = np.degrees(e_), []
    E = Mdeg + estar * np.sin(np.radians(Mdeg))
    for _ in range(4):
        Eit.append(E)
        E = E - (E - estar * np.sin(np.radians(E)) - Mdeg) / (1 - e_ * np.cos(np.radians(E)))
    Eit.append(E)
    kepler = {"a": a_, "e": e_, "M": Mdeg, "E_it": Eit, "E": solve_kepler(Mdeg, e_)}
    ellipse = []
    for n in ("Mercurio", "Venus", "Tierra", "Marte", "Júpiter", "Saturno", "Urano", "Neptuno", "Plutón"):
        a1, e1, q1, Q1 = orbit_elements(n, when)
        ellipse.append((n, a1, e1, q1, Q1, a1 * e1, 100 * (1 - np.sqrt(1 - e1 ** 2))))

    ans = {}
    ans["earth_mm"] = schwarzschild_mass_length(5.97217e24) * 1e3
    ans["jup_limb"] = 4 * schwarzschild_mass_length(1.89813e27) / 7.1492e7 * ARCSEC
    ans["dil_us"] = (1 - time_dilation([6.3781366e6, 0, 0], [_earth_body()])) * 86400 * 1e6
    ans["shapiro_km"] = 4 * Msun * np.arcsinh(AU / R_SUN) / 1e3
    ans["kepler_E"] = solve_kepler(57.2957795, 0.1)       # M = 1 rad
    q_pl = orbit_elements("Plutón", when)[2]
    a_ne = orbit_elements("Neptuno", when)[0]
    ans["pluto_q"], ans["nep_a"] = q_pl, a_ne
    return Data(when, gm, sun_limb, schw, mercury, table, dilation, mc, kepler, ellipse, ans)


def _sun_body():
    from .scene import Body
    return Body("Sol", M_SUN, np.zeros(3), radius=R_SUN)


def _earth_body():
    from .scene import Body
    return Body("Tierra", 5.97217e24, np.zeros(3), radius=6.3781366e6)


def _fdir(res):
    p = res.sol.y[3:6, -1]
    return p / np.linalg.norm(p)


# ------------------------------------------------------------------ figuras (matplotlib)
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans", "axes.grid": True,
                         "grid.alpha": 0.25, "figure.dpi": 100})
    return plt


def _png(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.06)
    fig.clf()
    buf.seek(0)
    return buf


def fig_flow():
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.2, 2.6))
    ax.axis("off")
    steps = ["Datos\n(masas, posiciones,\nincertidumbres)", "Unidades\ngeométricas\nG = c = 1", "Métrica\nSchwarzschild o\ncampo débil",
             "Geodésicas\n(luz y masas):\nODE", "Resultados\nángulos, retrasos,\nrelojes", "Mapa 3D\ny animación"]
    cols = ["#cfe8ff", "#d7f5dd", "#fff0c2", "#ffd9d2", "#e8d9ff", "#d9f2f2"]
    for i, (s, c) in enumerate(zip(steps, cols)):
        x = i * 1.62
        ax.add_patch(plt.Rectangle((x, 0.2), 1.38, 1.6, fc=c, ec="#334155", lw=1.2, zorder=2))
        ax.text(x + 0.69, 1.0, s, ha="center", va="center", fontsize=8.2, zorder=3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + 1.62, 1.0), xytext=(x + 1.38, 1.0), arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#334155"})
    ax.text(4.86, -0.15, "Monte Carlo: repetir con masas/posiciones perturbadas → incertidumbre", ha="center", fontsize=8, style="italic")
    ax.set_xlim(-0.1, 9.9)
    ax.set_ylim(-0.4, 2.0)
    return _png(fig)


def fig_lensing():
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.axis("off")
    ax.add_patch(plt.Circle((0, 0), 0.42, color="#FFD60A", ec="#92400e", zorder=3))
    ax.text(0, -0.78, "masa M", ha="center", fontsize=9)
    ax.plot([-3.2, -0.0], [1.0, 1.0], color="#0f172a", lw=2)
    ax.plot([0.0, 3.2], [1.0, 1.0 + 3.2 * np.tan(np.radians(9))], color="#0f172a", lw=2)
    ax.plot([0.0, 3.2], [1.0, 1.0], color="#64748b", lw=1.2, ls="--")
    ax.annotate("", xy=(0, 0.45), xytext=(0, 1.0), arrowprops={"arrowstyle": "<->", "lw": 1.2, "color": "#b91c1c"})
    ax.text(0.12, 0.7, "b", color="#b91c1c", fontsize=11)
    ax.text(2.0, 1.55, r"$\alpha \approx 4GM/(c^2 b)$", fontsize=11, color="#0f172a")
    ax.text(-2.4, 1.2, "rayo de luz", fontsize=9)
    ax.text(2.0, 0.82, "dirección sin masa", fontsize=8, color="#64748b")
    ax.set_xlim(-3.3, 3.4)
    ax.set_ylim(-1.0, 2.2)
    return _png(fig)


def fig_flamm():
    plt = _plt()
    fig = plt.figure(figsize=(6.4, 3.0))
    ax = fig.add_subplot(121)
    r = np.linspace(2.0, 14, 300)
    ax.plot(r, 2 * np.sqrt(2 * (r - 2)), color="#4338ca", lw=2)
    ax.axvline(2, color="#b91c1c", ls="--", lw=1)
    ax.text(2.2, 6.6, "horizonte\nr = 2M", color="#b91c1c", fontsize=8)
    ax.set_xlabel("r / M")
    ax.set_ylabel("z / M")
    ax.set_title(r"$z(r)=2\sqrt{2M(r-2M)}$", fontsize=10)
    ax3 = fig.add_subplot(122, projection="3d")
    rr, ph = np.meshgrid(np.linspace(2.0, 12, 40), np.linspace(0, 2 * np.pi, 60))
    ax3.plot_surface(rr * np.cos(ph), rr * np.sin(ph), 2 * np.sqrt(2 * (rr - 2)), cmap="viridis", linewidth=0, alpha=0.9)
    ax3.set_axis_off()
    ax3.view_init(elev=25, azim=-60)
    ax3.set_title("paraboloide de Flamm", fontsize=10)
    return _png(fig)


def fig_deflection():
    plt = _plt()
    bs = np.logspace(1.3, 4.0, 14)
    num = np.array([integrate_null(1.0, r0=max(1e5 * b, 1e6), b=b)[1] for b in bs])
    f1 = 4 / bs
    f2 = 4 / bs + 15 * np.pi / 4 / bs ** 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    a1.loglog(bs, f1, "--", color="#64748b", label=r"$4M/b$ (campo débil)")
    a1.loglog(bs, f2, "-", color="#4338ca", label=r"$4M/b + 15\pi M^2/4b^2$")
    a1.loglog(bs, num, "o", color="#b91c1c", ms=4, label="integración numérica")
    a1.set_xlabel("b / M")
    a1.set_ylabel("deflexión (rad)")
    a1.legend(fontsize=7)
    a2.semilogx(bs, (num / f1 - 1) * 100, "o-", color="#64748b", ms=3, label="vs. primer orden")
    a2.semilogx(bs, (num / f2 - 1) * 100, "o-", color="#4338ca", ms=3, label="vs. segundo orden")
    a2.set_xlabel("b / M")
    a2.set_ylabel("diferencia (%)")
    a2.legend(fontsize=7)
    fig.tight_layout()
    return _png(fig)


def fig_orbits(when):
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.6))
    cols = {"Mercurio": "#9ca3af", "Venus": "#f4a261", "Tierra": "#38bdf8", "Marte": "#ef4444",
            "Júpiter": "#c2410c", "Saturno": "#d6b88a", "Urano": "#2dd4bf", "Neptuno": "#6366f1", "Plutón": "#d946ef"}
    th = np.linspace(0, 2 * np.pi, 200)
    for ax, names, lim, ttl in ((a1, ("Mercurio", "Venus", "Tierra", "Marte"), 1.9, "Sistema interior"),
                                (a2, ("Júpiter", "Saturno", "Urano", "Neptuno", "Plutón"), 52, "Sistema exterior")):
        if ax is a1:
            ax.fill(np.concatenate([BELT_INNER_AU * np.cos(th), BELT_OUTER_AU * np.cos(th[::-1])]),
                    np.concatenate([BELT_INNER_AU * np.sin(th), BELT_OUTER_AU * np.sin(th[::-1])]), color="#8b7e74", alpha=0.35)
            ax.text(1.0, 1.75, "cinturón →", fontsize=7, color="#6b5f56")
        for n in names:
            p = orbit_path(n, when) / AU
            ax.plot(p[:, 0], p[:, 1], color=cols[n], lw=1.6, label=n)
            _q, _Q = orbit_elements(n, when)[2:]
        ax.plot(0, 0, "o", color="#FFD60A", mec="#92400e", ms=9, zorder=5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("x (UA)")
        ax.legend(fontsize=6.5, loc="lower left")
    a1.set_xlim(-1.9, 1.9)
    a1.set_ylim(-1.9, 1.9)
    fig.tight_layout()
    return _png(fig)


def fig_ellipse():
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    a, e = 1.0, 0.6
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(a * np.cos(th) - a * e, a * np.sqrt(1 - e * e) * np.sin(th), color="#4338ca", lw=2)
    ax.plot(0, 0, "o", color="#FFD60A", mec="#92400e", ms=11)
    ax.plot(-2 * a * e, 0, "x", color="#64748b")
    ax.plot(-a * e, 0, ".", color="#64748b")
    ax.annotate("Sol (foco)", (0, 0), (0.05, 0.25), fontsize=8)
    ax.annotate("perihelio q = a(1−e)", (a * (1 - e), 0), (0.45, -0.55), fontsize=8, arrowprops={"arrowstyle": "->"})
    ax.annotate("afelio Q = a(1+e)", (-a * (1 + e), 0), (-1.55, 0.45), fontsize=8, arrowprops={"arrowstyle": "->"})
    ax.text(-a * e - 0.02, -0.12, "centro", fontsize=7, color="#64748b", ha="center")
    ax.set_aspect("equal")
    ax.set_title("elipse con e = 0,6 (exagerada)", fontsize=9)
    ax.axis("off")
    return _png(fig)


_EQ_CACHE = {}


def eq_png(tex, fs=14):
    key = (tex, fs)
    if key in _EQ_CACHE:
        return _EQ_CACHE[key]
    plt = _plt()
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, f"${tex}$", fontsize=fs)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=260, bbox_inches="tight", pad_inches=0.04, transparent=True)
    plt.close(fig)
    buf.seek(0)
    _EQ_CACHE[key] = buf.getvalue()
    return _EQ_CACHE[key]


# ------------------------------------------------------------------ documento (reportlab)
def _fmt(x, d=4):
    return f"{x:.{d}g}".replace(".", ",")


def build_pdf(path, nivel="profesor", when=None, log=print):
    import matplotlib
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        KeepTogether,
        ListFlowable,
        ListItem,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    prof = nivel == "profesor"
    when = when or datetime.now(timezone.utc)
    D = compute(when, log)

    fd = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    for nm, fn in (("DV", "DejaVuSans.ttf"), ("DV-B", "DejaVuSans-Bold.ttf"), ("DV-I", "DejaVuSans-Oblique.ttf"),
                   ("DV-BI", "DejaVuSans-BoldOblique.ttf"), ("DVM", "DejaVuSansMono.ttf")):
        pdfmetrics.registerFont(TTFont(nm, os.path.join(fd, fn)))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

    INK = colors.HexColor("#0f172a")
    ACC = colors.HexColor("#4338ca")
    S = {
        "title": ParagraphStyle("title", fontName="DV-B", fontSize=26, leading=32, textColor=ACC, alignment=TA_CENTER, spaceAfter=10),
        "sub": ParagraphStyle("sub", fontName="DV", fontSize=13, leading=18, alignment=TA_CENTER, textColor=INK, spaceAfter=6),
        "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=16, leading=20, textColor=ACC, spaceBefore=14, spaceAfter=6),
        "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=12.2, leading=16, textColor=INK, spaceBefore=9, spaceAfter=3),
        "p": ParagraphStyle("p", fontName="DV", fontSize=9.6, leading=14, alignment=TA_JUSTIFY, spaceAfter=5, textColor=INK),
        "small": ParagraphStyle("small", fontName="DV", fontSize=8, leading=11, textColor=colors.HexColor("#475569")),
        "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=8.2, leading=11, alignment=TA_CENTER, textColor=colors.HexColor("#475569"), spaceAfter=8),
        "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.2, leading=10.5, textColor=INK),
        "code": ParagraphStyle("code", fontName="DVM", fontSize=8.3, leading=11, backColor=colors.HexColor("#f1f5f9"), borderPadding=5, spaceAfter=7, leftIndent=6),
    }
    story = []

    def H1(t): story.append(Paragraph(t, S["h1"]))
    def H2(t): story.append(Paragraph(t, S["h2"]))
    def P(t): story.append(Paragraph(t, S["p"]))
    def CODE(t): story.append(Paragraph(t.replace("\n", "<br/>").replace(" ", "&nbsp;"), S["code"]))
    def SP(h=6): story.append(Spacer(1, h))

    def EQ(tex, fs=14, num=None):
        png = eq_png(tex, fs)
        im = PILImage.open(io.BytesIO(png))
        w, h = im.size
        k = 72.0 / 260.0
        img = Image(io.BytesIO(png), width=w * k, height=h * k)
        if num:
            t = Table([[img, Paragraph(f"({num})", S["small"])]], colWidths=[15.0 * cm, 1.5 * cm])
            t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ALIGN", (0, 0), (0, 0), "CENTER")]))
            story.extend([Spacer(1, 3), t, Spacer(1, 3)])
        else:
            img.hAlign = "CENTER"
            story.extend([Spacer(1, 3), img, Spacer(1, 3)])

    def FIG(buf, width_cm, caption):
        im = PILImage.open(io.BytesIO(buf.getvalue()))
        w, h = im.size
        wp = width_cm * cm
        img = Image(io.BytesIO(buf.getvalue()), width=wp, height=wp * h / w)
        img.hAlign = "CENTER"
        story.append(KeepTogether([img, Paragraph(caption, S["cap"])]))

    def BOX(title, body_lines, fill="#eef2ff", edge="#4338ca"):
        cont = [Paragraph(f"<b>{title}</b>", S["cell"])] + [Paragraph(x, S["cell"]) for x in body_lines]
        t = Table([[cont]], colWidths=[16.4 * cm])
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(fill)), ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(edge)),
                               ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                               ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
        story.extend([t, Spacer(1, 7)])

    def TAB(rows, widths, head=True, align_right_from=1):
        data = [[Paragraph(str(c), S["cell"]) for c in r] for r in rows]
        t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if head else 0)
        st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if head:
            st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff"))]
        t.setStyle(TableStyle(st))
        story.extend([KeepTogether([t]), Spacer(1, 8)])

    def BUL(items):
        story.append(ListFlowable([ListItem(Paragraph(x, S["p"]), leftIndent=12) for x in items], bulletType="bullet", start="•", leftIndent=12))
        SP(3)

    # ---------------------------------------------------------------- portada
    story.append(Spacer(1, 3.2 * cm))
    story.append(Paragraph("relmap", S["title"]))
    story.append(Paragraph("Guía pedagógica", S["title"]))
    story.append(Paragraph("Cómo se resuelven las ecuaciones de campo de la relatividad general<br/>para dibujar el entorno de una nave y navegar por él", S["sub"]))
    story.append(Spacer(1, 1.0 * cm))
    story.append(Paragraph(f"<b>Nivel: {'PROFESOR (derivaciones, soluciones y sugerencias de aula)' if prof else 'ESTUDIANTE'}</b>", S["sub"]))
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph(f"Escena de ejemplo: Sistema Solar a fecha {when:%Y-%m-%d} (UTC)<br/>Versión de relmap {__version__} · generado el {datetime.now(timezone.utc):%Y-%m-%d}", S["sub"]))
    story.append(Spacer(1, 2.2 * cm))
    story.append(Paragraph("© 2026 Pedro Sordo Martínez · Licencia AGPL-3.0-or-later<br/>"
                           "Todos los números de este documento se calculan con el propio programa en el momento de generarlo.", S["small"]))
    story.append(PageBreak())

    # ---------------------------------------------------------------- 1
    H1("1. ¿Qué problema resuelve este programa?")
    P("Imagina una nave en el espacio. Tienes estimaciones de dónde están las estrellas, los planetas y sus masas. Quieres saber "
      "<b>cómo se curvan los rayos de luz</b>, <b>cuánto se retrasan las señales</b>, <b>cuánto atrasan los relojes</b> y <b>dónde hay masa que "
      "deforma el entorno</b>. La relatividad general responde a esto con una idea: la masa curva el espacio-tiempo, y los objetos libres "
      "(la luz incluida) siguen las líneas más rectas posibles en ese espacio curvo, llamadas <b>geodésicas</b>.")
    EQ(r"G_{\mu\nu}=\frac{8\pi G}{c^{4}}\,T_{\mu\nu}", 16, "1")
    P("Esta es la ecuación de campo de Einstein: a la izquierda, la geometría (curvatura); a la derecha, la materia y la energía. "
      "Wheeler lo resumió así: <i>la materia le dice al espacio-tiempo cómo curvarse; el espacio-tiempo le dice a la materia cómo moverse</i>.")
    H2("¿Por qué no basta un «solucionador universal»?")
    P("La ecuación (1) son <b>10 ecuaciones diferenciales parciales no lineales y acopladas</b>. No existe una receta general que, dadas "
      "unas posiciones, devuelva la solución exacta. En la práctica se usan tres niveles de aproximación; relmap implementa los dos primeros:")
    TAB([["Nivel", "Cuándo vale", "En relmap"],
         ["1. Solución exacta de un cuerpo (Schwarzschild)", "Una masa esférica, incluso con campo fuerte", "<b>Sí</b> (Fase 1): <font name='DVM'>integrate_null</font>, <font name='DVM'>integrate_timelike</font>"],
         ["2. Campo débil, varios cuerpos", "|Φ| ≪ 1 (sistemas planetarios, estrellas)", "<b>Sí</b> (Fases 2-4): <font name='DVM'>trace_ray</font>"],
         ["3. Relatividad numérica", "Campo fuerte, dos agujeros negros que se fusionan", "No (requiere superordenadores)"]],
        [5.6, 5.4, 5.4])
    FIG(fig_flow(), 16.0, "Figura 1. Flujo del método: de los datos estimados al mapa. La rama de Monte Carlo propaga las incertidumbres.")

    # ---------------------------------------------------------------- 2
    H1("2. Unidades geométricas: G = c = 1")
    P("En relatividad conviene medir la masa en metros. Se define la <b>masa geométrica</b> <i>M = G·m/c²</i>. Con ella, G y c desaparecen "
      "de las fórmulas y los cálculos numéricos quedan mejor condicionados.")
    EQ(r"M=\frac{G\,m}{c^{2}}\qquad\Longrightarrow\qquad r_s=2M=\frac{2Gm}{c^{2}}", 15, "2")
    TAB([["Cuerpo", "Masa (kg)", "M = Gm/c²"]] + [[n, f"{m:.4g}".replace(".", ","), (f"{g:.5g} m" if g >= 0.1 else f"{g*1e3:.4g} mm") .replace(".", ",") if g >= 1e-3 else f"{g*1e6:.4g} µm"] for n, m, g in D.gm],
        [4, 5.4, 5.4])
    P("Campo débil significa que <i>M/r ≪ 1</i> en todo el recorrido. En la superficie del Sol M/R ≈ 2,1·10⁻⁶; en la de la Tierra ≈ 7·10⁻¹⁰; "
      "un agujero negro tendría M/r ≈ 1/2. Por eso la aproximación de campo débil es excelente en el Sistema Solar.")
    if prof:
        BOX("Nota para el profesor", [
            ("Restituir unidades: una longitud en metros se convierte en tiempo dividiendo por c; una masa geométrica en kg se obtiene multiplicando por c²/G. "
            "Un error típico del alumnado es olvidar que, con G = c = 1, «masa», «longitud» y «tiempo» comparten unidad.")])

    # ---------------------------------------------------------------- 3
    H1("3. Un cuerpo: la solución exacta de Schwarzschild")
    H2("3.1 La métrica")
    P("Para una masa esférica M en reposo, la solución exacta de (1) en el vacío es la métrica de Schwarzschild (coordenadas t, r, θ, φ):")
    EQ(r"ds^{2}=-\left(1-\frac{2M}{r}\right)dt^{2}+\left(1-\frac{2M}{r}\right)^{-1}dr^{2}+r^{2}\left(d\theta^{2}+\sin^{2}\theta\,d\varphi^{2}\right)", 13, "3")
    P("La métrica dice cómo medir distancias y tiempos. Lejos (r → ∞) se reduce a la geometría plana de la relatividad especial.")
    H2("3.2 Las geodésicas: cómo se mueve la luz y las masas de prueba")
    EQ(r"\frac{d^{2}x^{\mu}}{d\lambda^{2}}+\Gamma^{\mu}_{\ \alpha\beta}\,\frac{dx^{\alpha}}{d\lambda}\frac{dx^{\beta}}{d\lambda}=0", 14, "4")
    P("Es la «línea más recta» en el espacio curvo. λ es un parámetro afín (para masas es el tiempo propio; para la luz, uno cualquiera). "
      "Los símbolos de Christoffel Γ se calculan a partir de la métrica. En el plano ecuatorial (θ = π/2) los no nulos son:")
    EQ(r"\Gamma^{t}_{tr}=\frac{M}{r(r-2M)},\ \ \Gamma^{r}_{tt}=\frac{M(r-2M)}{r^{3}},\ \ \Gamma^{r}_{rr}=-\frac{M}{r(r-2M)},\ \ \Gamma^{r}_{\varphi\varphi}=-(r-2M),\ \ \Gamma^{\varphi}_{r\varphi}=\frac{1}{r}", 11, "5")
    P("Sustituyendo en (4) (con punto = d/dλ) salen las tres ecuaciones que integra el programa:")
    EQ(r"\ddot t=-\frac{2M}{r(r-2M)}\dot t\,\dot r,\qquad \ddot r=-\frac{M(r-2M)}{r^{3}}\dot t^{2}+\frac{M}{r(r-2M)}\dot r^{2}+(r-2M)\dot\varphi^{2},\qquad \ddot\varphi=-\frac{2}{r}\dot r\,\dot\varphi", 11, "6")
    H2("3.3 Cantidades conservadas (una comprobación gratis)")
    P("Como la métrica no depende de t ni de φ, se conservan la energía E y el momento angular L. Con ellas la ecuación radial queda en forma «de energía»:")
    EQ(r"E=\left(1-\frac{2M}{r}\right)\dot t,\quad L=r^{2}\dot\varphi,\qquad \dot r^{2}=E^{2}-\left(1-\frac{2M}{r}\right)\left(\kappa+\frac{L^{2}}{r^{2}}\right)", 12, "7")
    P("con κ = 0 para la luz y κ = 1 para una masa de prueba. El parámetro de impacto de un rayo es b = L/E. "
      "El programa usa (7) para fijar las condiciones iniciales y puede comprobarse que E y L se conservan durante la integración.")
    H2("3.4 Resultados clásicos que el programa reproduce")
    P(f"<b>Deflexión de la luz.</b> Un rayo con parámetro de impacto b ≫ M se desvía un ángulo <i>α ≈ 4M/b</i> (el doble del valor newtoniano 2M/b). "
      f"Para un rayo que roza el Sol (b = 1,001 R☉), el programa da <b>{_fmt(D.sun_limb['num'],5)}″</b>; la fórmula 4M/b da {_fmt(D.sun_limb['formula'],5)}″ y la "
      f"predicción newtoniana sería {_fmt(D.sun_limb['newton'],4)}″. La observación del eclipse de 1919 (Dyson, Eddington y colaboradores) favoreció el valor relativista.")
    P(f"<b>Precisión del campo débil.</b> Con M = 1 y b = 10⁴: la integración exacta de Schwarzschild da α = {D.schw['exact']:.6e} rad; "
      f"la fórmula de primer orden 4M/b = {D.schw['first']:.6e}; con el término de segundo orden 4M/b + 15πM²/(4b²) = {D.schw['second']:.6e}.")
    FIG(fig_deflection(), 16.0, "Figura 2. Deflexión de la luz en función del parámetro de impacto (M = 1). Los puntos son la integración numérica de las ecuaciones (6).")
    P(f"<b>Precesión del perihelio de Mercurio.</b> Una órbita ligada no se cierra: el perihelio avanza Δφ = 6πM/[a(1−e²)] por vuelta. Integrando numéricamente la órbita de "
      f"Mercurio (a = 0,3871 UA, e = 0,2056) se obtiene {_fmt(D.mercury['num'],5)}″ por siglo; la fórmula da {_fmt(D.mercury['ana'],5)}″ por siglo. "
      "Es la parte relativista del famoso avance de unos 43″ por siglo que la mecánica de Newton no explicaba.")
    H2("3.5 El «embudo» del mapa: el paraboloide de Flamm")
    P("Para visualizar la curvatura espacial de Schwarzschild se «incrusta» en 3D la superficie ecuatorial (t = cte, θ = π/2). La superficie resultante es un paraboloide de revolución:")
    EQ(r"z(r)=2\sqrt{2M\,(r-2M)}\qquad (r\geq 2M)", 14, "8")
    FIG(fig_flamm(), 13.5, "Figura 3. Diagrama de incrustación de Flamm. En relmap esta idea inspira la «malla deformada» del mapa.")
    BOX("Importante (honestidad científica)", [
        ("En el mapa del Sistema Solar la <b>profundidad de los pozos es ilustrativa</b> (proporcional al logaritmo de la masa): si se dibujara a escala real, los planetas serían invisibles junto al Sol. "
        "Los números físicos se leen en las tablas, no en la profundidad.")], fill="#fff7ed", edge="#c2410c")

    # ---------------------------------------------------------------- 4
    H1("4. Varios cuerpos: campo débil")
    H2("4.1 Linealizar y sumar")
    P("Cuando la curvatura es pequeña, la métrica es la plana más una perturbación pequeña. La ecuación (1) se vuelve <b>lineal</b> en esa perturbación y las soluciones se <b>suman</b>: "
      "el potencial de varios cuerpos es la suma de los de cada uno (principio de superposición). En gauge isótropo:")
    EQ(r"ds^{2}=-(1+2\Phi)\,dt^{2}+(1-2\Phi)\,\delta_{ij}\,dx^{i}dx^{j},\qquad \Phi(\mathbf{x})=-\sum_{i}\frac{M_{i}}{|\mathbf{x}-\mathbf{x}_{i}|}", 14, "9")
    P("Dentro de un cuerpo esférico uniforme de radio R el potencial es el de la esfera maciza (teorema de Newton): Φ = −M(3R² − r²)/(2R³).")
    H2("4.2 La luz como en un medio con índice de refracción")
    P("En la métrica (9) los rayos de luz cumplen la ecuación de rayo de la óptica geométrica, con un «índice de refracción» n = 1 − 2Φ (algo mayor que 1 cerca de la masa):")
    EQ(r"\frac{d}{ds}\!\left(n\,\frac{d\mathbf{x}}{ds}\right)=\nabla n,\qquad n=1-2\Phi", 14, "10")
    P("Esto convierte un problema de relatividad en uno de óptica: la luz se curva hacia donde n es mayor, es decir, hacia las masas. "
      "El programa integra (10) con las variables (x, p = n·dx/ds), y acumula dos cantidades útiles por el camino:")
    EQ(r"\text{retraso}\;\;\Delta t=\frac{1}{c}\int (n-1)\,ds,\qquad\qquad \frac{d\tau}{dt}=\sqrt{1+2\Phi}\approx 1+\Phi", 13, "11")
    P(f"La primera es el <b>retraso de Shapiro</b> (la luz tarda más cerca de una masa); la segunda, la <b>dilatación gravitatoria del tiempo</b> de un reloj en reposo. "
      f"Para un rayo recto que pasa a distancia b de una masa M y recorre desde −a hasta +a, el retraso (en metros) es 4M·arsinh(a/b). "
      f"Ejemplo: de 1 UA a 1 UA rozando el Sol, {D.answers['shapiro_km']:.1f} km, es decir {D.answers['shapiro_km']*1e3/C*1e6:.0f} µs.")
    if prof:
        BOX("Derivación breve (profesor)", [
            ("De la geodésica nula ds² = 0 en (9): (1+2Φ)c²dt² = (1−2Φ)dl², luego la velocidad coordenada es c/n con n ≈ 1−2Φ. "
            "El principio de Fermat (tiempo mínimo, ∫n dl estacionario) da la ecuación de rayo (10). "
            "Con n = 1 + 2M/r y un rayo recto a distancia b, la desviación es α = ∫|∇⊥n| ds = ∫ 2Mb/(b²+x²)^{3/2} dx = 4M/b. "
            "La mitad procede de la curvatura temporal (g₀₀) y la otra mitad de la espacial (g_ij): "
            "de ahí el factor 2 respecto al cálculo newtoniano (2M/b).")])
    H2("4.3 Qué pasa si el rayo choca con un cuerpo")
    P("El programa define sucesos de parada: si el rayo entra en el radio de un cuerpo, la integración se detiene y se marca «IMPACTA». "
      "Esto evita resultados sin sentido físico (la luz no atraviesa planetas).")

    # ---------------------------------------------------------------- 5
    H1("5. Integración numérica")
    P("Las ecuaciones (6) y (10) son ecuaciones diferenciales ordinarias. Se escriben como un sistema de primer orden y = f(s, y) (posición, momento y, para (10), el retraso acumulado). "
      "Se resuelven con el método de Runge-Kutta <b>DOP853</b> (Dormand-Prince de orden 8, paso adaptativo) con tolerancia relativa 10⁻¹². "
      "Un paso adaptativo hace pasos pequeños donde la solución cambia rápido (cerca de una masa) y grandes donde es suave.")
    BUL(["<b>Eventos</b>: detectan impactos y capturas por el horizonte.",
         "<b>Salida densa</b>: permite muestrear el camino con tantos puntos como se quiera sin recalcular.",
         ("<b>Verificación</b>: las predicciones se contrastan con fórmulas analíticas conocidas (deflexión, Shapiro, precesión, simetría, superposición). "
         "El repositorio incluye una batería de pruebas automáticas que lo comprueba.")])
    H2("5.1 Por qué el mapa exagera la curvatura de los rayos")
    P("Las desviaciones de la luz son de microradianes (o menos): dibujadas a escala serían rectas indistinguibles. En el mapa, la curvatura del rayo se <b>exagera</b> por un factor K que se escribe en la leyenda. "
      "Para no amplificar errores numéricos, el tramo visible se reconstruye integrando las <i>direcciones</i> (que se calculan con gran precisión) y no las posiciones.")

    # ---------------------------------------------------------------- 6
    H1("6. Datos estimados → incertidumbre (Monte Carlo)")
    P("En una nave, las masas y posiciones son estimaciones. El método de Monte Carlo repite el cálculo muchas veces con masas y posiciones perturbadas al azar según sus incertidumbres, "
      "y resume el resultado (mediana e intervalo del 90 %).")
    P(f"<b>Ejemplo 1.</b> Un rayo a 1,5 R☉ del Sol con la masa del Sol conocida al 1 %: la deflexión resulta {_fmt(D.mc['sol'][0],4)}″ ± {_fmt(D.mc['sol'][1],2)}″ (un 1 %, porque la deflexión es proporcional a la masa).")
    rows = [["Error en la posición de Júpiter", "Deflexión media (″)", "Desviación típica (″)"]]
    for sig, m, sd in D.mc["jup"]:
        rows.append([("ninguno" if sig == 0 else f"{sig:,.0f} km".replace(",", " ")), _fmt(m, 4), _fmt(sd, 3)])
    P("<b>Ejemplo 2.</b> Un rayo que pasa a 2 radios de Júpiter. Aquí lo decisivo no es la masa sino <i>dónde está</i> el planeta:")
    TAB(rows, [7.0, 4.7, 4.7])
    P("Cuando el error de posición es mayor que el parámetro de impacto del rayo, el efecto de lente de ese cuerpo se pierde en el ruido: es una lección real para la navegación.")

    # ---------------------------------------------------------------- 7
    H1("7. Colocar los cuerpos: órbitas elípticas y el Sistema Solar")
    P("Para dibujar el Sistema Solar hay que saber dónde está cada planeta en una fecha. relmap usa los <b>elementos keplerianos aproximados</b> de JPL (Standish; válidos 1800-2050): "
      "seis números por planeta que varían linealmente con el tiempo (a = semieje, e = excentricidad, I = inclinación, L = longitud media, ϖ = longitud del perihelio, Ω = nodo ascendente).")
    EQ(r"p(T)=p_{0}+\dot p\,T,\qquad T=\frac{JD-2451545}{36525},\qquad M=L-\varpi,\qquad \omega=\varpi-\Omega", 13, "12")
    H2("7.1 La ecuación de Kepler")
    P("La posición en la elipse depende de la anomalía excéntrica E, que no tiene solución algebraica: se resuelve por iteración de Newton-Raphson.")
    EQ(r"M=E-e\sin E\qquad\Longrightarrow\qquad E_{k+1}=E_{k}-\frac{E_{k}-e\sin E_{k}-M}{1-e\cos E_{k}}", 14, "13")
    EQ(r"x'=a(\cos E-e),\qquad y'=a\sqrt{1-e^{2}}\,\sin E", 14, "14")
    P("(x', y') están en el plano de la órbita, con el Sol en un foco; se giran con ω, I y Ω al plano de la eclíptica. "
      f"<b>Ejemplo (Marte, {when:%Y-%m-%d}):</b> a = {_fmt(D.kepler['a'],6)} UA, e = {_fmt(D.kepler['e'],5)}, M = {_fmt(D.kepler['M'],5)}°. "
      "Iterando (13) desde E₀ = M + e*·sin M: " + " → ".join(f"{x:.6f}°" for x in D.kepler["E_it"]) + ". Converge en 3-4 pasos.")
    H2("7.2 ¿Por qué las órbitas parecen casi circulares?")
    P("Porque lo son <i>casi</i>. Una elipse con excentricidad e tiene semieje menor b = a√(1−e²). Para la Tierra (e = 0,0167) el eje menor es solo un 0,014 % más corto que el mayor: "
      "indistinguible a simple vista. Lo que sí se aprecia es que el Sol no está en el centro sino en un foco, desplazado a·e (para la Tierra, unos 2,5 millones de km).")
    rows = [["Cuerpo", "a (UA)", "e", "perihelio q (UA)", "afelio Q (UA)", "desplaz. a·e (UA)", "achat. (%)"]]
    for n, a1, e1, q1, Q1, ae, fl in D.ellipse:
        rows.append([n, _fmt(a1, 5), _fmt(e1, 3), _fmt(q1, 5), _fmt(Q1, 5), _fmt(ae, 3), _fmt(fl, 3)])
    TAB(rows, [2.3, 2.1, 1.9, 2.6, 2.5, 2.6, 2.4])
    P("Mercurio (e ≈ 0,21) y Plutón (e ≈ 0,25) son claramente elípticos. El perihelio de Plutón (≈ 29,7 UA) queda <b>dentro</b> de la órbita media de Neptuno (≈ 30,1 UA) durante parte de su recorrido.")
    FIG(fig_orbits(when), 16.0, "Figura 4. Órbitas elípticas reales a la fecha elegida (vista cenital). A la izquierda, el sistema interior y el cinturón de asteroides; a la derecha, el exterior con Plutón.")
    FIG(fig_ellipse(), 8.4, "Figura 5. Geometría de la elipse: el Sol ocupa un foco, no el centro.")
    H2("7.3 La Luna, Plutón y el cinturón de asteroides")
    BUL([("<b>Tierra y Luna.</b> Los elementos dan la posición del baricentro Tierra-Luna; la Tierra se desplaza de él ≈ 4 670 km hacia el lado opuesto a la Luna. "
         "La Luna se modela como órbita circular de 384 400 km con inclinación de 5,145° (error de decenas de miles de km)."),
         "<b>Plutón.</b> JPL retiró a Plutón de su tabla de elementos tras 2006; se usan los valores de la tabla original de Standish, reproducidos en bibliotecas abiertas (pykep, satkit). Masa y radio: JPL (planeta enano).",
         (f"<b>Cinturón de asteroides.</b> Se marca entre {BELT_INNER_AU:g} y {BELT_OUTER_AU:g} UA, entre Marte y Júpiter. Es una nube <i>ilustrativa</i> (con los huecos de Kirkwood, que son resonancias con Júpiter), no asteroides reales. "
         "Es la frontera habitual entre el sistema solar <b>interior</b> (planetas rocosos) y el <b>exterior</b> (gigantes y objetos transneptunianos). "
         "La masa total del cinturón es menos de una milésima de la de la Tierra (≈ 4 % de la masa lunar), de modo que no afecta a los cálculos de lente.")])
    TAB([["Cuerpo", "Error nominal de posición (JPL, 1800-2050)"],
         ["Mercurio / Venus / Tierra", "≈ 1 000 / 4 000 / 6 000 km"], ["Marte", "≈ 25 000 km"], ["Júpiter / Saturno", "≈ 600 000 / 1 500 000 km"],
         ["Urano / Neptuno", "≈ 1 000 000 / 200 000 km"], ["Plutón", "no publicado; usamos 30 000 000 km (estimación gruesa)"], ["Luna", "decenas de miles de km (modelo simple)"]],
        [6.0, 10.4])

    # ---------------------------------------------------------------- 8
    H1(f"8. Resultados del Sistema Solar ({when:%Y-%m-%d})")
    P("Un rayo rasante a cada cuerpo (en el plano de la eclíptica, a 2 radios del centro). «Total» = todo el sistema actuando sobre el rayo; «aporte» = lo que añade ese cuerpo (total − mismo rayo sin él). "
      "Para cuerpos pequeños el total lo domina el campo del Sol a lo largo del trayecto.")
    rows = [["Rayo rasante a…", "Total (″)", "Aporte del cuerpo (″)", "Fórmula 4GM/(bc²) (″)", "Retraso (µs)"]]
    for r in D.table:
        rows.append([r["name"], f"{r['total']:.4g}".replace(".", ","), f"{r['own']:.4g}".replace(".", ","), f"{r['formula']:.4g}".replace(".", ","), f"{r['delay_us']:.4g}".replace(".", ",")])
    TAB(rows, [4.0, 2.6, 3.8, 3.8, 2.2])
    P("La columna «aporte» coincide con la fórmula analítica: el método de superposición funciona. Dilatación temporal en la superficie (fracción en que un reloj atrasa respecto a uno lejano):")
    TAB([["Cuerpo", "1 − dτ/dt"]] + [[n, f"{v:.3e}".replace(".", ",")] for n, v in D.dilation], [5, 5])

    # ---------------------------------------------------------------- 9
    H1("9. Ejercicios" + (" con soluciones" if prof else ""))
    ex = [
        ("Convierte la masa de la Tierra (5,97·10²⁴ kg) a masa geométrica M = Gm/c².", f"M ≈ {D.answers['earth_mm']:.3g} mm."),
        ("Calcula la deflexión de un rayo que roza el limbo de Júpiter (R = 71 492 km, m = 1,898·10²⁷ kg).", f"α = 4GM/(c²R) ≈ {D.answers['jup_limb']:.4f}″ (unas 100 veces menos que el Sol)."),
        ("¿Cuántos microsegundos al día atrasa un reloj en el ecuador terrestre respecto a uno muy lejano? (Usa dτ/dt ≈ 1 + Φ.)", f"≈ {D.answers['dil_us']:.0f} µs al día."),
        ("Resuelve la ecuación de Kepler E − e·sin E = M para M = 1 rad y e = 0,1, con dos iteraciones de Newton partiendo de E₀ = M.", f"E ≈ {np.radians(D.answers['kepler_E']):.5f} rad (converge en 2-3 pasos)."),
        ("Explica sin fórmulas por qué la órbita de la Tierra «parece» circular aunque sea una elipse.", "Su excentricidad es 0,0167: el eje menor es solo un 0,014 % menor que el mayor; lo único notable es el desplazamiento del Sol del centro (foco)."),
        ("¿Puede Plutón estar más cerca del Sol que Neptuno? Compara q de Plutón con a de Neptuno.", f"Sí: q(Plutón) ≈ {D.answers['pluto_q']:.1f} UA < a(Neptuno) ≈ {D.answers['nep_a']:.1f} UA (no chocan porque sus órbitas están inclinadas y en resonancia 3:2)."),
        ("¿Por qué el programa exagera los rayos en el mapa pero no en las tablas?", "Porque el mapa es visual (ángulos de 10⁻⁶ rad serían invisibles) y las tablas son los números reales. El factor K se indica siempre."),
        ("Si la posición de un planeta tiene un error de 600 000 km, ¿puedes predecir su efecto de lente sobre un rayo que pasa a 140 000 km de su centro?", "No: el error de posición es mayor que el parámetro de impacto; el efecto se pierde en el ruido (ver sección 6)."),
    ]
    for i, (q, a) in enumerate(ex, 1):
        story.append(Paragraph(f"<b>{i}.</b> {q}", S["p"]))
        if prof:
            story.append(Paragraph(f"<font color='#166534'><b>Solución:</b> {a}</font>", S["p"]))
    if prof:
        H1("10. Sugerencias de aula y errores frecuentes")
        H2("Secuencia sugerida (3 sesiones de 50 min)")
        BUL(["<b>Sesión 1 — Idea física.</b> Secciones 1-2 y el mapa: ver las vistas «interior» y «exterior» y el cinturón. Pregunta guía: ¿qué significa que la masa curva el espacio?",
             "<b>Sesión 2 — Geodésicas y luz.</b> Secciones 3-4. Que el alumnado reproduzca 1,75″ con el programa y compare con el valor newtoniano.",
             "<b>Sesión 3 — Datos reales.</b> Secciones 6-7. Animar el movimiento de traslación, discutir excentricidades y propagar incertidumbres con Monte Carlo."])
        H2("Errores conceptuales frecuentes")
        BUL(["«La gravedad es una fuerza». En relatividad general los objetos libres siguen geodésicas; la «fuerza» es geometría.",
             "Confundir la <b>malla ilustrativa</b> con la curvatura real: la profundidad dibujada no es a escala.",
             "Creer que la luz «tiene masa» y por eso se curva: se curva porque el espacio-tiempo es curvo.",
             ("Pensar que el cinturón de asteroides es un planeta destruido: la masa total es mucho menor que la de la Luna y su composición es muy diversa. "
             "(La hipótesis de Olbers de un planeta «Faetón» es histórica y está descartada.)"),
             "Olvidar que el resultado depende de la <b>incertidumbre en la posición</b>, no solo de la masa."])
        H2("Ampliaciones")
        BUL(["Relatividad numérica (formalismo 3+1) para campos fuertes: <i>no</i> está en relmap.",
             "Comparar con efemérides de alta precisión (JPL Horizons) para cuantificar el error de los elementos aproximados.",
             "Añadir más cuerpos (Ceres, lunas de los gigantes) con elementos de la base de datos de cuerpos menores de JPL."])
    else:
        H1("10. Para seguir aprendiendo")
        BUL(["Cambia la fecha del Sistema Solar con <font name='DVM'>relmap-solar --date 1900-01-01</font> y observa cómo cambian las posiciones.",
             "Pulsa ▶ en el mapa y sigue a la Luna con la vista «Centrar en Luna».",
             "Edita un archivo de escena YAML y mira qué pasa si duplicas la masa de Júpiter."])

    # ---------------------------------------------------------------- 11 límites
    H1("11. Límites del programa")
    BUL(["<b>Campo débil</b>: no vale cerca de horizontes de agujeros negros ni para campos fuertes; para un cuerpo en campo fuerte usa la solución de Schwarzschild.",
         "Los cuerpos se tratan como <b>estáticos mientras la luz pasa</b> (la luz cruza el Sistema Solar en horas).",
         "Las posiciones son <b>aproximadas</b> (elementos medios de JPL, 1800-2050); no sustituyen a las efemérides de alta precisión.",
         "No se incluyen aún planetas enanos distintos de Plutón, asteroides reales ni lunas de otros planetas.",
         "La malla del mapa es <b>ilustrativa</b>; la curvatura de los rayos se exagera por un factor indicado."])

    H1("12. Glosario breve")
    TAB([["Término", "Significado"],
         ["Geodésica", "Trayectoria «más recta» posible en un espacio-tiempo curvo."],
         ["Métrica", "Regla que da distancias y tiempos entre puntos cercanos del espacio-tiempo."],
         ["Parámetro de impacto (b)", "Distancia a la que pasaría el rayo de la masa si no se desviara."],
         ["Retraso de Shapiro", "Retraso adicional de una señal que pasa cerca de una masa."],
         ["Excentricidad (e)", "Cuánto se aleja una órbita de ser un círculo (0 = círculo)."],
         ["Perihelio / afelio", "Punto de la órbita más cercano / más lejano al Sol."],
         ["Cinturón de asteroides", "Región entre Marte y Júpiter con millones de cuerpos menores."]], [5.0, 11.4])

    H1("13. Referencias")
    BUL(["Einstein, A. (1915). <i>Die Feldgleichungen der Gravitation</i>. Sitzungsberichte der Preußischen Akademie der Wissenschaften.",
         "Misner, Thorne y Wheeler (1973). <i>Gravitation</i>. W. H. Freeman.",
         "Carroll, S. (2004). <i>Spacetime and Geometry</i>. Addison-Wesley.",
         "Shapiro, I. I. (1964). Fourth test of general relativity. <i>Phys. Rev. Lett.</i> 13, 789.",
         "Standish, E. M. Keplerian elements for approximate positions of the major planets. JPL Solar System Dynamics (ssd.jpl.nasa.gov/planets/approx_pos.html).",
         "JPL Planetary Physical Parameters (ssd.jpl.nasa.gov/planets/phys_par.html).",
         "Dormand, J. R. y Prince, P. J. (1980). A family of embedded Runge-Kutta formulae. <i>J. Comp. Appl. Math.</i> 6, 19."])
    H1("14. Cómo reproducir todo esto")
    CODE("pip install -e \".[dev]\"\npytest                                   # comprueba la física\nrelmap-solar --date 2026-11-05 --offline  # mapa interactivo\nrelmap-pdf --nivel ambos                  # esta guía")

    def _foot(c, doc):
        c.saveState()
        c.setFont("DV", 8)
        c.setFillColor(colors.HexColor("#64748b"))
        c.drawString(2 * cm, 1.1 * cm, f"relmap — Guía pedagógica ({'profesor' if prof else 'estudiante'})")
        c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
        c.restoreState()

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title=f"relmap — Guía pedagógica ({nivel})", author="Pedro Sordo Martínez")
    doc.build(story, onFirstPage=_foot, onLaterPages=_foot)
    return path
