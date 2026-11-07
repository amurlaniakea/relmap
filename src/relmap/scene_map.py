# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Pedro Sordo Martínez
"""Mapa 3D interactivo y animado de una escena.

* Vistas (menú): 1 Todo el sistema · 2 Sistema solar interior · 3 Sistema solar exterior (la
  frontera es el cinturón de asteroides) y «Centrar en <cuerpo>» para acercarte a cada uno.
* Animación: los planetas recorren sus órbitas elípticas (movimiento de traslación). Se puede
  pausar, cambiar la velocidad, ir al revés y saltar a cualquier fecha (1800-2050).
* Colores: el amarillo es solo del Sol. Modo oscuro (por defecto) / claro con interruptor.
* La malla es ILUSTRATIVA (profundidad ~ log masa): muestra dónde hay masa, no cuánto. Los
  números físicos salen en la tabla de la terminal. La curvatura de los rayos (microradianes) se
  exagera por un factor K que se indica en la leyenda.

La geometría se calcula en el navegador (JavaScript) con las mismas fórmulas que
``relmap.solar_system``; así el movimiento es continuo sin pre-calcular fotogramas.
"""
from __future__ import annotations

import json
import unicodedata

import numpy as np
import plotly.graph_objects as go

AU = 1.495978707e11
SUN_YELLOW = "#FFD60A"
PALETTE = {
    "sol": SUN_YELLOW, "sun": SUN_YELLOW,
    "mercurio": "#A8A29E", "mercury": "#A8A29E",
    "venus": "#F4A261",
    "tierra": "#38BDF8", "earth": "#38BDF8",
    "luna": "#CBD5E1", "moon": "#CBD5E1",
    "marte": "#EF4444", "mars": "#EF4444",
    "jupiter": "#C2410C",
    "saturno": "#D6B88A", "saturn": "#D6B88A",
    "urano": "#2DD4BF", "uranus": "#2DD4BF",
    "neptuno": "#6366F1", "neptune": "#6366F1",
    "pluton": "#F0ABFC", "pluto": "#F0ABFC",
}
FALLBACK = ["#A78BFA", "#F472B6", "#34D399", "#FB7185", "#60A5FA", "#F97316", "#14B8A6", "#C084FC"]
BELT_COLOR = "#8B7E74"

THEMES = {
    "dark": {"bg": "#0b1020", "text": "#e5e7eb", "axis": "#94a3b8", "grid": "rgba(148,163,184,0.30)",
             "ray": "#f8fafc", "outline": "rgba(0,0,0,0.75)", "menu": "#1e293b", "border": "#475569",
             "active": "#4338ca", "belt": "rgba(160,148,138,0.75)"},
    "light": {"bg": "#f8fafc", "text": "#0f172a", "axis": "#475569", "grid": "rgba(51,65,85,0.55)",
              "ray": "#0f172a", "outline": "#1f2937", "menu": "#e2e8f0", "border": "#94a3b8",
              "active": "#c7d2fe", "belt": "rgba(110,98,88,0.80)"},
}


def _norm(name: str) -> str:
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()


def body_colors(bodies):
    out, i = {}, 0
    for b in bodies:
        c = b.color or PALETTE.get(_norm(b.name))
        if c is None:
            c = FALLBACK[i % len(FALLBACK)]
            i += 1
        out[b.name] = c
    return out


def _rgba(hexc: str, a: float) -> str:
    h = hexc.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def default_views(bodies, results, belt=None, orbits=None):
    """Vistas del mapa. ``belt`` = dict(inner_m, outer_m) activa «interior» y «exterior».

    Cada vista: name, group, include (nombres o None = todos), center (m), follow (cuerpo o None),
    h (semianchura de la caja en m), belt (bool).
    """
    heavy = max(bodies, key=lambda b: b.mass_kg)
    c = heavy.position.astype(float)
    pts = [b.position[:2] for b in bodies]
    for name, o in (orbits or []):
        o = np.asarray(o, float)
        pts += [o[:, :2].min(axis=0), o[:, :2].max(axis=0)]
    for r in results:
        p = r.sol.y[:3]
        pts += [p[:2, 0], p[:2, -1]]
    pts = np.array(pts)
    h_all = 1.12 * float(np.max(np.abs(pts - c[:2]))) or 1.0
    views = [{"name": "1 · Todo el sistema", "group": "Vistas", "include": None, "center": c,
              "follow": None, "h": h_all, "belt": belt is not None}]
    if belt is not None:
        dist = {b.name: float(np.linalg.norm(b.position - heavy.position)) for b in bodies}
        inner = [b.name for b in bodies if b is heavy or dist[b.name] <= belt["inner_m"]]
        outer = [b.name for b in bodies if b is heavy or dist[b.name] >= belt["outer_m"]]
        views.append({"name": "2 · Sistema solar interior", "group": "Vistas", "include": inner,
                      "center": c, "follow": None, "h": 1.1 * belt["outer_m"], "belt": True})
        ext = [0.0]
        for name, o in (orbits or []):
            if name in outer:
                ext.append(float(np.max(np.abs(np.asarray(o, float)[:, :2] - c[:2]))))
        for b in bodies:
            if b.name in outer:
                ext.append(float(np.max(np.abs(b.position[:2] - c[:2]))))
        views.append({"name": "3 · Sistema solar exterior", "group": "Vistas", "include": outer,
                      "center": c, "follow": None, "h": 1.12 * max(ext), "belt": True})
    for b in bodies:
        others = [np.linalg.norm(o.position - b.position) for o in bodies if o is not b]
        dn = min(others) if others else np.inf
        R = b.radius if b.radius > 0 else 0.0
        h = (max(60 * R, 1.25 * dn) if dn <= 2000 * R else 60 * R) if R > 0 else 0.02 * h_all
        views.append({"name": f"Centrar en {b.name}", "group": "Centrar en", "include": None,
                      "center": b.position.astype(float), "follow": b.name, "h": float(h), "belt": False})
    return views


def _ray_world(results, views):
    """Por rayo y vista: tramo visible (coordenadas del mundo, km) con curvatura exagerada."""
    data = []
    for r in results:
        sg = np.linspace(0.0, r.s_end, 6000)
        y = r.sol.sol(sg)
        d = y[3:6].T
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        data.append((r, y[:3].T, d, sg))
    out = []
    for r, path, dirs, sg in data:
        per_view = {}
        for vi, v in enumerate(views):
            c, h = v["center"], v["h"]
            ok = (np.abs(path[:, 0] - c[0]) <= h * 1.02) & (np.abs(path[:, 1] - c[1]) <= h * 1.02)
            idx = np.flatnonzero(ok)
            if len(idx) < 8:
                continue
            i0, i1 = idx[0], idx[-1]
            ds = np.diff(sg[i0:i1 + 1])
            seg_len = float(ds.sum())
            if seg_len < 0.4 * 2 * h:
                continue
            dd = dirs[i0:i1 + 1]
            seg = path[i0] + np.vstack([np.zeros(3), np.cumsum(0.5 * (dd[1:] + dd[:-1]) * ds[:, None], axis=0)])
            chord = seg[-1] - seg[0]
            dc = chord / np.linalg.norm(chord)
            base = seg[0] + np.outer((seg - seg[0]) @ dc, dc)
            dev = seg - base
            maxdev = float(np.abs(dev).max())
            K = min(0.08 * seg_len / maxdev, 1e14) if maxdev > 1e-16 * seg_len else 1.0
            show = base + K * dev
            per_view[vi] = {"K": K, "pts": np.round(show / 1e3, 1).ravel().tolist()}
        out.append({"name": r.name, "hit": r.hit, "views": per_view})
    return out


_JS = r"""
(function(){
  var gd = document.getElementById('{plot_id}');
  var CFG = __CFG__, THEMES = __THEMES__, LAYOUT = __LAYOUT__;
  var AU = 1.495978707e11, D2R = Math.PI / 180;
  document.body.style.margin = '0';
  gd.style.height = '100vh';

  /* ---------- efemérides (mismas fórmulas que relmap.solar_system) ---------- */
  function solveKepler(Mdeg, e){
    var es = e / D2R, M = (((Mdeg + 180) % 360) + 360) % 360 - 180, E = M + es * Math.sin(M * D2R);
    for (var k = 0; k < 100; k++){
      var dM = M - (E - es * Math.sin(E * D2R)), dE = dM / (1 - e * Math.cos(E * D2R));
      E += dE; if (Math.abs(dE) <= 1e-10) break;
    }
    return E;
  }
  function helio(key, jd){
    var el = CFG.eph.elements[key], T = (jd - 2451545.0) / 36525.0, v = [];
    for (var i = 0; i < 6; i++) v.push(el[0][i] + el[1][i] * T);
    var a = v[0], e = v[1], I = v[2], L = v[3], wbar = v[4], Om = v[5];
    var omega = wbar - Om, M = L - wbar, E = solveKepler(M, e) * D2R;
    var xp = a * (Math.cos(E) - e), yp = a * Math.sqrt(1 - e * e) * Math.sin(E);
    var co = Math.cos(omega * D2R), so = Math.sin(omega * D2R), cO = Math.cos(Om * D2R), sO = Math.sin(Om * D2R);
    var cI = Math.cos(I * D2R), sI = Math.sin(I * D2R);
    return [((co*cO - so*sO*cI) * xp + (-so*cO - co*sO*cI) * yp) * AU,
            ((co*sO + so*cO*cI) * xp + (-so*sO + co*cO*cI) * yp) * AU,
            ((so*sI) * xp + (co*sI) * yp) * AU];
  }
  function moonOffset(jd){
    var T = (jd - 2451545.0) / 36525.0, M = CFG.eph.moon;
    var lam = (218.3164477 + 481267.88123421 * T) * D2R, node = (125.0445479 - 1934.1362891 * T) * D2R, u = lam - node;
    var cu = Math.cos(u), su = Math.sin(u), cn = Math.cos(node), sn = Math.sin(node), ci = Math.cos(M.inc), si = Math.sin(M.inc);
    return [M.dist * (cu*cn - su*sn*ci), M.dist * (cu*sn + su*cn*ci), M.dist * su * si];
  }
  function moonRing(jd, center, n){
    var T = (jd - 2451545.0) / 36525.0, M = CFG.eph.moon, node = (125.0445479 - 1934.1362891 * T) * D2R;
    var ci = Math.cos(M.inc), si = Math.sin(M.inc), cn = Math.cos(node), sn = Math.sin(node), out = [];
    for (var k = 0; k < n; k++){
      var u = 2 * Math.PI * k / (n - 1), cu = Math.cos(u), su = Math.sin(u);
      out.push([center[0] + M.dist * (cu*cn - su*sn*ci), center[1] + M.dist * (cu*sn + su*cn*ci), center[2] + M.dist * su * si]);
    }
    return out;
  }
  function positions(jd){
    var P = {}, earth = null;
    CFG.bodies.forEach(function(b){
      if (CFG.dynamic){
        if (b.key === 'sol') { P[b.name] = [0, 0, 0]; return; }
        if (b.key === 'tierra' || b.key === 'luna') { return; }
        if (CFG.eph.elements[b.key]) { P[b.name] = helio(b.key, jd); return; }
      }
      P[b.name] = b.pos.slice();
    });
    if (CFG.dynamic && CFG.earthName){
      var emb = helio('embary', jd), rel = moonOffset(jd), f = CFG.eph.moon.ratio;
      var e = [emb[0] - rel[0]*f, emb[1] - rel[1]*f, emb[2] - rel[2]*f];
      P[CFG.earthName] = e;
      if (CFG.moonName) P[CFG.moonName] = [e[0] + rel[0], e[1] + rel[1], e[2] + rel[2]];
    }
    return P;
  }

  /* ---------- malla ilustrativa: pozos por cuerpo, profundidad ~ log masa ---------- */
  function makeSurf(P, h){
    var lm = CFG.bodies.map(function(b){ return Math.log10(Math.max(b.mass, 1e-30)); });
    var mn = Math.min.apply(null, lm), mx = Math.max.apply(null, lm), L = 0.35 * h, D = [], W = [], X = [], Y = [];
    CFG.bodies.forEach(function(b, i){
      var s = (mx - mn < 1e-9) ? 0.5 : (lm[i] - mn) / (mx - mn);
      D.push(L * (0.12 + 0.88 * s)); W.push(h * (0.10 + 0.15 * s)); X.push(P[b.name][0]); Y.push(P[b.name][1]);
    });
    var n = D.length;
    return {L: L, f: function(x, y){
      var z = 0;
      for (var i = 0; i < n; i++){ var dx = x - X[i], dy = y - Y[i]; z -= D[i] * W[i] / Math.sqrt(dx*dx + dy*dy + W[i]*W[i]); }
      return z;
    }};
  }
  var GN = 27, GM = 80, gridCache = {};
  function gridPattern(h){
    var k = String(h); if (gridCache[k]) return gridCache[k];
    var gx = [], gy = [], i, j, v, t;
    for (i = 0; i < GN; i++){ v = -h + 2*h*i/(GN-1); for (j = 0; j < GM; j++){ t = -h + 2*h*j/(GM-1); gx.push(t); gy.push(v); } gx.push(NaN); gy.push(NaN); }
    for (i = 0; i < GN; i++){ v = -h + 2*h*i/(GN-1); for (j = 0; j < GM; j++){ t = -h + 2*h*j/(GM-1); gx.push(v); gy.push(t); } gx.push(NaN); gy.push(NaN); }
    return gridCache[k] = {x: gx, y: gy};
  }

  /* ---------- índices de trazas ---------- */
  var IDX = {grid: -1, belt: -1, apsis: -1, orbit: {}, body: {}, ray: []};
  gd.data.forEach(function(t, i){
    var m = t.meta || {};
    if (m.role === 'grid') IDX.grid = i; else if (m.role === 'belt') IDX.belt = i; else if (m.role === 'apsis') IDX.apsis = i;
    else if (m.role === 'orbit') IDX.orbit[m.key] = i; else if (m.role === 'body') IDX.body[m.key] = i; else if (m.role === 'ray') IDX.ray[m.k] = i;
  });

  var state = {view: 0, jd: CFG.jd0, playing: false, speed: 30.4375, dir: 1, zr: null};
  function inc(V, name){ return !V.include || V.include.indexOf(name) >= 0; }

  function render(fresh){
    var V = CFG.views[state.view], P = positions(state.jd), h = V.h;
    var c = (V.follow && P[V.follow]) ? P[V.follow] : V.center, u = (h >= 0.02 * AU) ? AU : 1000, ul = (u === AU) ? 'UA' : 'km';
    var S = makeSurf(P, h), ids = [], X = [], Y = [], Z = [], VIS = [], zmin = 0, lim = h * 1.02;
    function ins(x, y){ return Math.abs(x - c[0]) <= lim && Math.abs(y - c[1]) <= lim; }
    function put(i, x, y, z, vis){ ids.push(i); X.push(x); Y.push(y); Z.push(z); VIS.push(vis); }
    function poly(pts, vis){ var x = [], y = [], z = [];
      pts.forEach(function(p){ if (ins(p[0], p[1])){ x.push((p[0]-c[0])/u); y.push((p[1]-c[1])/u); var zz = S.f(p[0], p[1]) / u; z.push(zz); if (zz < zmin) zmin = zz; }
                                else { x.push(NaN); y.push(NaN); z.push(NaN); } });
      return [x, y, z, vis]; }
    // malla
    var G = gridPattern(h), gx = [], gy = [], gz = [];
    for (var i = 0; i < G.x.length; i++){
      if (isNaN(G.x[i])) { gx.push(NaN); gy.push(NaN); gz.push(NaN); continue; }
      var wx = c[0] + G.x[i], wy = c[1] + G.y[i], zz = S.f(wx, wy) / u;
      gx.push(G.x[i] / u); gy.push(G.y[i] / u); gz.push(zz); if (zz < zmin) zmin = zz;
    }
    put(IDX.grid, gx, gy, gz, true);
    // órbitas (elipses; la Luna es un anillo alrededor de la Tierra)
    Object.keys(IDX.orbit).forEach(function(nm){
      var pts;
      if (CFG.dynamic && nm === CFG.moonName && P[CFG.earthName]) pts = moonRing(state.jd, P[CFG.earthName], 120);
      else { var f = CFG.orbits[nm].pts; pts = []; for (var k = 0; k < f.length; k += 3) pts.push([f[k]*AU, f[k+1]*AU, f[k+2]*AU]); }
      var r = poly(pts, inc(V, nm)); put(IDX.orbit[nm], r[0], r[1], r[2], r[3]);
    });
    // cinturón
    if (IDX.belt >= 0){ var bf = CFG.belt, bp = []; for (var k2 = 0; k2 < bf.length; k2 += 3) bp.push([bf[k2]*AU, bf[k2+1]*AU, bf[k2+2]*AU]);
      var rb = poly(bp, !!V.belt); put(IDX.belt, rb[0], rb[1], rb[2], rb[3]); }
    // perihelios y afelios
    if (IDX.apsis >= 0){ var ap = [], vis = [];
      CFG.apsides.forEach(function(a){ ap.push(inc(V, a.name) ? [a.pos[0]*AU, a.pos[1]*AU, a.pos[2]*AU] : [1e30, 1e30, 0]); });
      var ra = poly(ap, true); put(IDX.apsis, ra[0], ra[1], ra[2], true); }
    // cuerpos
    CFG.bodies.forEach(function(b){
      var p = P[b.name], v = inc(V, b.name), ok = ins(p[0], p[1]);
      var zz = S.f(p[0], p[1]) / u; if (ok && zz < zmin) zmin = zz;
      put(IDX.body[b.name], [ok ? (p[0]-c[0])/u : NaN], [ok ? (p[1]-c[1])/u : NaN], [ok ? zz : NaN], v && ok);
    });
    // rayos: solo valen en la fecha inicial
    var atT0 = Math.abs(state.jd - CFG.jd0) < 1e-6;
    CFG.rays.forEach(function(r, ri){
      var d = r.views[state.view], show = atT0 && !!d;
      if (show){ var x = [], y = [], z = [];
        for (var k = 0; k < d.pts.length; k += 3){ var wx = d.pts[k]*1e3, wy = d.pts[k+1]*1e3;
          x.push((wx-c[0])/u); y.push((wy-c[1])/u); var zz = S.f(wx, wy)/u; z.push(zz); if (zz < zmin) zmin = zz; }
        put(IDX.ray[ri], x, y, z, true);
      } else put(IDX.ray[ri], [NaN], [NaN], [NaN], false);
    });
    var lay = null;
    if (fresh || !state.zr){
      var zlo = 1.2 * zmin, zhi = 0.12 * S.L / u; if (zlo > -1e-9) zlo = -0.1 * h / u;
      var asp = Math.max(0.18, Math.min(0.6, (zhi - zlo) / (2 * h / u)));
      state.zr = [zlo, zhi]; state.asp = asp;
      lay = {'title.text': CFG.title + ' — ' + V.name,
        'scene.xaxis.range': [-h/u, h/u], 'scene.yaxis.range': [-h/u, h/u], 'scene.zaxis.range': [zlo, zhi],
        'scene.xaxis.title.text': 'Δx (' + ul + ')', 'scene.yaxis.title.text': 'Δy (' + ul + ')',
        'scene.aspectratio': {x: 1, y: 1, z: asp}};
      if (fresh) lay['scene.camera'] = {eye: {x: 1.5, y: -1.6, z: 0.9}};
    }
    var upd = {x: X, y: Y, z: Z, visible: VIS};
    var done = Plotly.update(gd, upd, lay || {}, ids);
    if (fresh){
      var nm = [], rid = [];
      CFG.rays.forEach(function(r, ri){ var d = r.views[state.view]; rid.push(IDX.ray[ri]); nm.push(r.name + (d ? ' (curvatura ×' + d.K.toPrecision(3) + ')' : '') + (r.hit ? ' — IMPACTA ' + r.hit : '')); });
      if (rid.length) Plotly.restyle(gd, {name: nm}, rid);
    }
    updLabel();
    return done;
  }

  /* ---------- tema ---------- */
  function apply(n){
    var T = THEMES[n], R = document.documentElement.style;
    document.body.style.background = T.bg;
    ['menu','active','text','border','axis'].forEach(function(k){ R.setProperty('--rm-' + k, T[k]); });
    Plotly.relayout(gd, LAYOUT[n]);
    var r = [], b = [];
    gd.data.forEach(function(t, i){ var m = t.meta || {}; if (m.role === 'ray') r.push(i); if (m.role === 'body') b.push(i); });
    Plotly.restyle(gd, {'line.color': T.grid}, [IDX.grid]);
    if (r.length) Plotly.restyle(gd, {'line.color': T.ray}, r);
    if (b.length) Plotly.restyle(gd, {'textfont.color': T.text, 'marker.line.color': T.outline}, b);
    if (IDX.belt >= 0) Plotly.restyle(gd, {'marker.color': T.belt}, [IDX.belt]);
    if (IDX.apsis >= 0) Plotly.restyle(gd, {'marker.line.color': T.text}, [IDX.apsis]);
    ico.textContent = (n === 'dark') ? '\u263E' : '\u2600'; txt.textContent = (n === 'dark') ? 'Modo oscuro' : 'Modo claro';
  }

  /* ---------- panel de control ---------- */
  var st = document.createElement('style');
  st.textContent =
    '#rm-panel{position:fixed;top:10px;left:14px;z-index:10;display:flex;flex-direction:column;gap:8px;font:13px system-ui,sans-serif;color:var(--rm-text);max-width:360px}' +
    '#rm-panel .row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}' +
    '#rm-panel .box{padding:6px 12px;border:1px solid var(--rm-border);border-radius:10px;background:var(--rm-menu)}' +
    '#rm-sw{display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none;width:max-content;border-radius:999px}' +
    '#rm-sw .tr{position:relative;width:38px;height:20px;border-radius:999px;background:#64748b;transition:.2s}' +
    '#rm-sw .tr::after{content:"";position:absolute;top:2px;left:2px;width:16px;height:16px;border-radius:50%;background:#fff;transition:.2s}' +
    '#rm-sw input{display:none}#rm-sw input:checked + .tr{background:#6366f1}#rm-sw input:checked + .tr::after{left:20px}' +
    '#rm-panel select,#rm-panel input[type=date],#rm-panel button{font:13px system-ui,sans-serif;color:var(--rm-text);background:var(--rm-menu);border:1px solid var(--rm-border);border-radius:8px;padding:4px 8px}' +
    '#rm-panel select{width:100%}#rm-panel button{cursor:pointer;min-width:34px}#rm-panel button:hover,#rm-panel select:hover{background:var(--rm-active)}' +
    '#rm-panel .lab{opacity:.8;font-size:12px}#rm-date{font-weight:600;font-variant-numeric:tabular-nums}';
  document.head.appendChild(st);
  var panel = document.createElement('div'); panel.id = 'rm-panel';
  function box(){ var d = document.createElement('div'); d.className = 'box'; panel.appendChild(d); return d; }
  // tema
  var b0 = box(), sw = document.createElement('label'); sw.id = 'rm-sw'; sw.title = 'Cambiar modo oscuro / claro';
  var ico = document.createElement('span'), txt = document.createElement('span'), inp = document.createElement('input'), tr = document.createElement('span');
  inp.type = 'checkbox'; inp.checked = true; tr.className = 'tr'; [ico, inp, tr, txt].forEach(function(e){ sw.appendChild(e); }); b0.appendChild(sw);
  inp.addEventListener('change', function(){ apply(inp.checked ? 'dark' : 'light'); });
  // vistas
  var b1 = box(), lv = document.createElement('div'); lv.className = 'lab'; lv.textContent = 'Vista'; b1.appendChild(lv);
  var sel = document.createElement('select'), groups = {};
  CFG.views.forEach(function(v, i){
    var g = groups[v.group]; if (!g){ g = document.createElement('optgroup'); g.label = v.group; groups[v.group] = g; sel.appendChild(g); }
    var o = document.createElement('option'); o.value = i; o.textContent = v.name; g.appendChild(o);
  });
  sel.addEventListener('change', function(){ state.view = +sel.value; state.zr = null; render(true); });
  b1.appendChild(sel);
  // tiempo
  var play, dateTxt, speedSel, dirBtn, dateInp, reset;
  function updLabel(){ if (!dateTxt) return; var d = new Date((state.jd - 2440587.5) * 86400000); dateTxt.textContent = d.toISOString().slice(0, 10); if (dateInp && document.activeElement !== dateInp) dateInp.value = dateTxt.textContent; }
  if (CFG.dynamic){
    var b2 = box(), lt = document.createElement('div'); lt.className = 'lab'; lt.textContent = 'Movimiento de traslación'; b2.appendChild(lt);
    var r1 = document.createElement('div'); r1.className = 'row'; b2.appendChild(r1);
    play = document.createElement('button'); play.textContent = '\u25B6'; play.title = 'Reproducir / pausar';
    dirBtn = document.createElement('button'); dirBtn.textContent = '\u21C4'; dirBtn.title = 'Invertir el sentido del tiempo';
    reset = document.createElement('button'); reset.textContent = '\u27F2'; reset.title = 'Volver a la fecha inicial';
    dateTxt = document.createElement('span'); dateTxt.id = 'rm-date';
    [play, dirBtn, reset, dateTxt].forEach(function(e){ r1.appendChild(e); });
    var r2 = document.createElement('div'); r2.className = 'row'; b2.appendChild(r2);
    speedSel = document.createElement('select'); speedSel.style.width = 'auto';
    [['1 día / s', 1], ['1 semana / s', 7], ['1 mes / s', 30.4375], ['6 meses / s', 182.625], ['1 año / s', 365.25], ['5 años / s', 1826.25], ['25 años / s', 9131.25], ['100 años / s', 36525]].forEach(function(s){
      var o = document.createElement('option'); o.value = s[1]; o.textContent = s[0]; if (s[1] === 30.4375) o.selected = true; speedSel.appendChild(o); });
    dateInp = document.createElement('input'); dateInp.type = 'date'; dateInp.min = '1800-01-01'; dateInp.max = '2050-12-31';
    r2.appendChild(speedSel); r2.appendChild(dateInp);
    var note = document.createElement('div'); note.className = 'lab'; note.textContent = 'Los rayos de luz solo se dibujan en la fecha inicial. Elementos JPL válidos 1800–2050.'; b2.appendChild(note);
    var busy = false, last = 0;
    function tick(ts){
      if (!state.playing) return;
      var dt = Math.min((ts - last) / 1000, 0.1); last = ts;
      if (!busy){
        busy = true; state.jd += state.dir * state.speed * dt;
        var lo = 2378496.5, hi = 2469807.5; if (state.jd < lo) state.jd = lo; if (state.jd > hi) state.jd = hi;
        var p = render(false); (p && p.then ? p : Promise.resolve()).then(function(){ busy = false; }, function(){ busy = false; });
      }
      requestAnimationFrame(tick);
    }
    play.addEventListener('click', function(){ state.playing = !state.playing; play.textContent = state.playing ? '\u23F8' : '\u25B6'; if (state.playing){ last = performance.now(); requestAnimationFrame(tick); } });
    dirBtn.addEventListener('click', function(){ state.dir *= -1; dirBtn.style.outline = state.dir < 0 ? '2px solid var(--rm-active)' : 'none'; });
    speedSel.addEventListener('change', function(){ state.speed = +speedSel.value; });
    reset.addEventListener('click', function(){ state.jd = CFG.jd0; render(false); });
    dateInp.addEventListener('change', function(){ if (!dateInp.value) return; state.jd = Date.parse(dateInp.value + 'T00:00:00Z') / 86400000 + 2440587.5; render(false); });
    window.__rm = {state: state, positions: positions, render: render};
  } else { window.__rm = {state: state, positions: positions, render: render}; }
  document.body.appendChild(panel);

  Plotly.Plots.resize(gd);
  apply('dark');
  render(true);
})();
"""


def _theme_layout(name):
    T = THEMES[name]
    lay = {"paper_bgcolor": T["bg"], "plot_bgcolor": T["bg"], "font.color": T["text"],
           "title.font.color": T["text"], "scene.bgcolor": T["bg"], "annotations[0].font.color": T["axis"]}
    for ax in "xyz":
        lay[f"scene.{ax}axis.color"] = T["axis"]
        lay[f"scene.{ax}axis.gridcolor"] = T["grid"]
        lay[f"scene.{ax}axis.zerolinecolor"] = T["grid"]
    return lay


def build_scene_map(scene, results, exaggerate="auto", out_html=None, title="relmap — escena",
                    orbits=None, offline=False, ephemeris=None, jd0=None, belt=None, apsides=None,
                    views=None, orbit_info=None):
    """Construye el mapa. ``ephemeris`` (relmap.solar_system.ephemeris_spec()) + ``jd0`` activan la
    animación; ``belt`` = dict(inner_m, outer_m, points (n,3) m); ``apsides`` = [(nombre, peri, afe)]."""
    bodies = list(scene.bodies)
    colors = body_colors(bodies)
    dark = THEMES["dark"]
    if views is None:
        views = default_views(bodies, results, belt=None if belt is None else
                              {"inner_m": belt["inner_m"], "outer_m": belt["outer_m"]}, orbits=orbits)
    masses = np.array([b.mass_kg for b in bodies], float)
    rank = np.argsort(np.argsort(-masses))
    sizes = {b.name: (20 if rank[i] == 0 else 7 + 5 * (1 - rank[i] / max(len(bodies) - 1, 1)))
             for i, b in enumerate(bodies)}
    dynamic = ephemeris is not None and jd0 is not None
    key_of = {b.name: _norm(b.name) for b in bodies}
    earth = next((b.name for b in bodies if key_of[b.name] in ("tierra", "earth")), None)
    moon = next((b.name for b in bodies if key_of[b.name] in ("luna", "moon")), None)
    if dynamic and (earth is None):
        moon = None
    orb = {}
    for name, pts in (orbits or []):
        pts = np.asarray(pts, float)
        info = (orbit_info or {}).get(name, "")
        orb[name] = {"pts": [] if (dynamic and name == moon) else np.round(pts / AU, 5).ravel().tolist(),
                     "info": info}
    aps = []
    for name, q, Q in (apsides or []):
        aps.append({"name": name, "kind": "perihelio", "pos": np.round(np.asarray(q) / AU, 5).tolist()})
        aps.append({"name": name, "kind": "afelio", "pos": np.round(np.asarray(Q) / AU, 5).tolist()})

    cfg = {
        "title": title, "dynamic": bool(dynamic), "jd0": float(jd0 or 0.0), "eph": ephemeris,
        "earthName": earth, "moonName": moon,
        "bodies": [{"name": b.name, "key": key_of[b.name], "mass": float(b.mass_kg),
                    "pos": [float(v) for v in b.position]} for b in bodies],
        "views": [{"name": v["name"], "group": v["group"], "include": v["include"],
                   "center": [float(x) for x in v["center"]], "follow": v["follow"], "h": float(v["h"]),
                   "belt": bool(v["belt"])} for v in views],
        "orbits": orb, "apsides": aps,
        "belt": [] if belt is None else np.round(np.asarray(belt["points"]) / AU, 4).ravel().tolist(),
        "rays": _ray_world(results, views),
    }

    fig = go.Figure()
    empty = dict(x=[0.0], y=[0.0], z=[0.0])
    fig.add_trace(go.Scatter3d(**empty, mode="lines", meta={"role": "grid"}, showlegend=False,
                               hoverinfo="skip", line=dict(color=dark["grid"], width=1)))
    for name in orb:
        info = orb[name]["info"]
        fig.add_trace(go.Scatter3d(**empty, mode="lines", meta={"role": "orbit", "key": name}, showlegend=False,
                                   hovertext=f"órbita de {name}" + (f" — {info}" if info else ""), hoverinfo="text",
                                   line=dict(color=_rgba(colors.get(name, "#94a3b8"), 0.7), width=2)))
    if belt is not None:
        fig.add_trace(go.Scatter3d(**empty, mode="markers", meta={"role": "belt"}, name="Cinturón de asteroides (ilustrativo)",
                                   hoverinfo="skip", marker=dict(size=1.6, color=dark["belt"])))
    if aps:
        fig.add_trace(go.Scatter3d(**empty, mode="markers", meta={"role": "apsis"}, name="◇ perihelio / afelio",
                                   hovertext=[f"{a['kind']} de {a['name']}" for a in aps], hoverinfo="text",
                                   marker=dict(size=5, symbol="diamond-open", color=[colors.get(a["name"], "#94a3b8") for a in aps],
                                               line=dict(color=dark["text"], width=2))))
    for b in bodies:
        fig.add_trace(go.Scatter3d(**empty, mode="markers+text", text=[b.name], textposition="top center", name=b.name,
                                   meta={"role": "body", "key": b.name}, hovertext=f"{b.name} — {b.mass_kg:.4g} kg",
                                   hoverinfo="text", textfont=dict(color=dark["text"], size=12),
                                   marker=dict(size=sizes[b.name], color=colors[b.name], line=dict(color=dark["outline"], width=1.5))))
    for ri, r in enumerate(results):
        fig.add_trace(go.Scatter3d(**empty, mode="lines", meta={"role": "ray", "k": ri}, name=r.name,
                                   line=dict(color=dark["ray"], width=5), visible=False))

    axis = dict(showbackground=False, color=dark["axis"], gridcolor=dark["grid"], zerolinecolor=dark["grid"])
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(color=dark["text"])),
        paper_bgcolor=dark["bg"], plot_bgcolor=dark["bg"], font=dict(color=dark["text"]),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", x=0.99, y=0.93, xanchor="right"),
        annotations=[dict(text="Malla ilustrativa (profundidad ~ log masa) · curvatura de los rayos exagerada ×K · "
                               "rueda = zoom, arrastrar = girar, menú = cambiar de vista",
                          x=0.5, y=0.0, xref="paper", yref="paper", showarrow=False,
                          font=dict(color=dark["axis"], size=11))],
        scene=dict(bgcolor=dark["bg"], aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.3),
                   xaxis=dict(title="Δx", **axis), yaxis=dict(title="Δy", **axis),
                   zaxis=dict(title="profundidad (ilustrativa)", showticklabels=False, **axis)))
    if out_html:
        js = (_JS.replace("__CFG__", json.dumps(cfg, allow_nan=False))
                 .replace("__THEMES__", json.dumps(THEMES))
                 .replace("__LAYOUT__", json.dumps({k: _theme_layout(k) for k in THEMES})))
        fig.write_html(out_html, include_plotlyjs=True if offline else "cdn", post_script=js,
                       config={"scrollZoom": True, "displaylogo": False})
    fig._relmap_cfg = cfg
    return fig
