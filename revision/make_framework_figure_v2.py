"""Figure 1 for the revised manuscript: the screen-confirm-replicate framework.

Hand-built SVG in the paper's schematic design system (Arial text, Georgia
italic symbols, #fcfbfa panels with dashed #ababab borders, variable colours:
wind #bcd1ea/#7d9cc4, humidity #d1e6cd/#8fae88, cloud #d4c2a1/#b8a37a,
temperature #f2ceb1/#dbab84, emphasis #c097a7/#8f5f73, validation #efe9b0/#d3c97e).
Rendered to PDF/PNG with headless Chrome (HTML wrapper with explicit charset).
Numbers are read from the three-stage outputs, not typed by hand.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "revision_outputs"
FIG = OUT / "paper_figures_20261002"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
W, H = 1200, 720
VAR = {"W": ("#bcd1ea", "#7d9cc4"), "H": ("#d1e6cd", "#8fae88"), "C": ("#d4c2a1", "#b8a37a"), "T": ("#f2ceb1", "#dbab84")}
EMPH = ("#e9dbe1", "#8f5f73")
VALID = ("#efe9b0", "#d3c97e")
INK, MUTED = "#222222", "#555555"


def t(x, y, s, size=13, weight="normal", anchor="middle", color=INK, italic=False, family="Arial"):
    style = "font-style:italic;" if italic else ""
    return f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}" style="{style}">{s}</text>'


def math(x, y, s, size=14, anchor="middle"):
    return t(x, y, s, size=size, anchor=anchor, italic=True, family="Georgia")


def panel(x, y, w, h, title, tag):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#fcfbfa" stroke="#ababab" stroke-width="1.3" stroke-dasharray="6 4"/>'
            f'<rect x="{x + 14}" y="{y + 12}" width="{w - 28}" height="30" rx="15" fill="{EMPH[0]}" stroke="{EMPH[1]}" stroke-width="1.1"/>'
            + t(x + w / 2, y + 32, f"{tag}  {title}", size=14.5, weight="bold"))


def node(cx, cy, key, r=17):
    f, s = VAR[key]
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{f}" stroke="{s}" stroke-width="1.6"/>' + math(cx, cy + 5, key, size=15)


def arrow(x1, y1, x2, y2, color="#6f6f6f", width=1.6, dash=None, marker="arr"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d} marker-end="url(#{marker})"/>'


def layer(x, y, key, label, sub):
    f, s = VAR[key]
    pts = f"{x},{y + 22} {x + 30},{y} {x + 130},{y} {x + 100},{y + 22}"
    grid = "".join(f'<line x1="{x + 30 + 24 * k}" y1="{y}" x2="{x + 24 * k}" y2="{y + 22}" stroke="{s}" stroke-width="0.7"/>' for k in range(1, 4))
    return (f'<polygon points="{pts}" fill="{f}" stroke="{s}" stroke-width="1.2"/>' + grid
            + math(x + 142, y + 16, label, size=15, anchor="start") + t(x + 160, y + 16, sub, size=10.5, anchor="start", color=MUTED))


def build():
    s3 = pd.read_csv(OUT / "three_stage" / "three_stage_summary.csv").set_index("edge_type").loc["ALL"]
    n = {k: f"{int(s3[k]):,}" for k in ("tested", "stage1_screened", "stage2_confirmed", "eval_wb2_replicated", "eval_wb2_negctrl_n", "eval_wb2_negctrl_replicated")}
    e = []
    e.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">')
    e.append('<defs><marker id="arr" markerUnits="userSpaceOnUse" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">'
             '<path d="M0,1 L10,6 L0,11 z" fill="#6f6f6f"/></marker>'
             '<marker id="big" markerUnits="userSpaceOnUse" markerWidth="18" markerHeight="18" refX="15" refY="9" orient="auto">'
             '<path d="M0,1 L16,9 L0,17 z" fill="#8f5f73"/></marker></defs>')
    e.append(f'<rect x="8" y="8" width="{W - 16}" height="{H - 16}" rx="16" fill="#fefefe" stroke="#b5b5b5" stroke-width="1.2" stroke-dasharray="6 4"/>')

    # Panel 1: data and regions
    x0, y0, pw, ph = 31, 26, 262, 392
    e.append(panel(x0, y0, pw, ph, "Regional data", ""))
    for k, (key, lab, sub) in enumerate((("T", "T", "850 hPa"), ("W", "W", "850 hPa |v|"), ("H", "H", "850 hPa q"), ("C", "C", "total cloud"))):
        e.append(layer(x0 + 22, y0 + 62 + 36 * k, key, lab, sub))
    e.append(t(x0 + pw / 2, y0 + 228, "ERA5 1979–2025 · 6-hourly · 64 × 32", size=11.5))
    e.append(arrow(x0 + pw / 2, y0 + 238, x0 + pw / 2, y0 + 262, color="#8a8a8a"))
    e.append(f'<rect x="{x0 + 34}" y="{y0 + 268}" width="{pw - 68}" height="30" rx="8" fill="white" stroke="#ababab"/>')
    e.append(t(x0 + pw / 2, y0 + 288, "66 regions × 4 variables", size=12.5, weight="bold"))
    e.append(t(x0 + pw / 2, y0 + 318, "anomalies fitted on 1979–2018 only", size=11, color=MUTED))
    e.append(t(x0 + pw / 2, y0 + 342, "+ u, v, qu, qv · six ERA5 controls", size=11, color=MUTED))
    e.append(t(x0 + pw / 2, y0 + 362, "+ CERES satellite cloud 2017–2025", size=11, color=MUTED))

    # Panel 2: screen
    x1 = x0 + pw + 30
    e.append(panel(x1, y0, pw, ph, "Screen", "Stage 1 ·"))
    cx, cy = x1 + pw / 2, y0 + 112
    pos = {"W": (cx - 78, cy - 22), "H": (cx, cy + 20), "C": (cx + 78, cy - 22), "T": (cx, cy - 52)}
    for a, b in (("W", "H"), ("H", "C"), ("W", "C"), ("T", "H"), ("T", "C")):
        (xa, ya), (xb, yb) = pos[a], pos[b]
        dx, dy = xb - xa, yb - ya
        L = (dx * dx + dy * dy) ** 0.5
        e.append(arrow(xa + dx / L * 19, ya + dy / L * 19, xb - dx / L * 21, yb - dy / L * 21))
    (xa, ya), (xb, yb) = pos["C"], pos["H"]
    e.append(f'<path d="M{xa - 8},{ya + 18} Q{cx + 58},{cy + 30} {xb + 20},{yb + 4}" fill="none" stroke="#b0616a" stroke-width="1.3" stroke-dasharray="4 3" marker-end="url(#arr)"/>')
    for k, p in pos.items():
        e.append(node(*p, k))
    e.append(t(cx, y0 + 176, "physics-guided candidates, k = 2 neighbours", size=11, color=MUTED))
    e.append(t(cx, y0 + 192, "dashed: reverse candidates (direction test)", size=10.5, color="#b0616a"))
    e.append(math(cx, y0 + 226, "y<tspan font-size='10' dy='4'>t</tspan><tspan dy='-4'> = α + γ</tspan><tspan font-size='10' dy='-6'>⊤</tspan><tspan dy='6'>c</tspan><tspan font-size='10' dy='4'>t</tspan><tspan dy='-4'> + β x</tspan><tspan font-size='10' dy='4'>t−ℓ</tspan>", size=15))
    e.append(t(cx, y0 + 250, "c = target's own lags 1–3 (recall)", size=11, color=MUTED))
    e.append(t(cx, y0 + 270, "calendar-time HAC · BH q &lt; 0.05", size=11, color=MUTED))
    e.append(f'<rect x="{x1 + 40}" y="{y0 + 300}" width="{pw - 80}" height="54" rx="10" fill="white" stroke="{EMPH[1]}" stroke-width="1.1"/>')
    e.append(t(cx, y0 + 324, f"{n['stage1_screened']} / {n['tested']}", size=17, weight="bold"))
    e.append(t(cx, y0 + 343, "edge–lag hypotheses, 1979–2018", size=10.5, color=MUTED))

    # Panel 3: confirm
    x2 = x1 + pw + 30
    e.append(panel(x2, y0, pw, ph, "Confirm", "Stage 2 ·"))
    cx2 = x2 + pw / 2
    gx, gy = cx2 - 66, y0 + 62
    for i in range(6):
        for j in range(8):
            key = "WHCT"[(i + j) % 4]
            e.append(f'<rect x="{gx + 17 * j}" y="{gy + 17 * i}" width="14" height="14" rx="2" fill="{VAR[key][0]}" stroke="{VAR[key][1]}" stroke-width="0.6"/>')
    e.append(t(cx2, y0 + 182, "dense VAR(3) on all 264 regional histories", size=11, color=MUTED))
    e.append(math(cx2, y0 + 226, "X<tspan font-size='10' dy='4'>t</tspan><tspan dy='-4'> = Σ</tspan><tspan font-size='10' dy='4'>ℓ≤3</tspan><tspan dy='-4'> A</tspan><tspan font-size='10' dy='4'>ℓ</tspan><tspan dy='-4'> X</tspan><tspan font-size='10' dy='4'>t−ℓ</tspan><tspan dy='-4'> + ε</tspan><tspan font-size='10' dy='4'>t</tspan>", size=15))
    e.append(t(cx2, y0 + 250, "removes indirect and shared-history edges", size=11, color=MUTED))
    e.append(t(cx2, y0 + 270, "HAC · BH q &lt; 0.05 (precision)", size=11, color=MUTED))
    e.append(f'<rect x="{x2 + 40}" y="{y0 + 300}" width="{pw - 80}" height="54" rx="10" fill="white" stroke="{EMPH[1]}" stroke-width="1.1"/>')
    e.append(t(cx2, y0 + 324, f"{n['stage2_confirmed']} confirmed", size=17, weight="bold"))
    e.append(t(cx2, y0 + 343, "screened and confirmed, 1979–2018", size=10.5, color=MUTED))

    # Panel 4: replicate
    x3 = x2 + pw + 30
    e.append(panel(x3, y0, pw, ph, "Replicate", "Stage 3 ·"))
    cx3 = x3 + pw / 2
    bx, by, bw = x3 + 22, y0 + 74, pw - 44
    segs = [(0.0, 0.70, "#dcdcdc", "1979–2018 discovery"), (0.70, 0.88, VAR["W"][0], "2019–2023"), (0.88, 1.0, "#f3e2c8", "2023–25")]
    for a, b, col, lab in segs:
        e.append(f'<rect x="{bx + a * bw}" y="{by}" width="{(b - a) * bw - 2}" height="26" rx="4" fill="{col}" stroke="#9a9a9a" stroke-width="0.8"/>')
    e.append(t(bx + 0.35 * bw, by + 18, "1979–2018 · fit", size=11))
    for k, (col, lab) in enumerate(((VAR["W"][0], "2019–2023 · held out, primary"), ("#f3e2c8", "2023–2025 · held out, second"))):
        e.append(f'<rect x="{bx + 6}" y="{by + 38 + 18 * k}" width="12" height="12" rx="2" fill="{col}" stroke="#9a9a9a" stroke-width="0.8"/>')
        e.append(t(bx + 24, by + 48 + 18 * k, lab, size=10.5, color=MUTED, anchor="start"))
    e.append(t(cx3, y0 + 168, "frozen preprocessing and selection", size=11, color=MUTED))
    e.append(t(cx3, y0 + 198, "same sign + one-sided HAC test,", size=11.5))
    e.append(t(cx3, y0 + 216, "BH within the confirmed family", size=11.5))
    e.append(t(cx3, y0 + 244, "frozen-coefficient predictive gain", size=11, color=MUTED))
    e.append(t(cx3, y0 + 264, "unconfirmed hypotheses = negative control", size=11, color=MUTED))
    e.append(f'<rect x="{x3 + 22}" y="{y0 + 300}" width="{pw - 44}" height="54" rx="10" fill="white" stroke="{EMPH[1]}" stroke-width="1.1"/>')
    e.append(t(cx3, y0 + 324, f"{n['eval_wb2_replicated']} / {n['stage2_confirmed']} replicate", size=17, weight="bold"))
    e.append(t(cx3, y0 + 343, f"control: {n['eval_wb2_negctrl_replicated']} / {n['eval_wb2_negctrl_n']} (2019–2023)", size=10.5, color=MUTED))

    for xa in (x0 + pw, x1 + pw, x2 + pw):
        e.append(f'<line x1="{xa + 3}" y1="{y0 + 210}" x2="{xa + 27}" y2="{y0 + 210}" stroke="#8f5f73" stroke-width="3" marker-end="url(#big)"/>')

    # Validation suite
    vy, vh = y0 + ph + 26, 238
    e.append(f'<rect x="26" y="{vy}" width="{W - 52}" height="{vh}" rx="12" fill="#fcfbfa" stroke="#ababab" stroke-width="1.3" stroke-dasharray="6 4"/>')
    e.append(f'<rect x="{W / 2 - 170}" y="{vy + 12}" width="340" height="30" rx="15" fill="{VALID[0]}" stroke="{VALID[1]}" stroke-width="1.1"/>')
    e.append(t(W / 2, vy + 32, "Validation suite around every stage", size=14.5, weight="bold"))
    tiles = [("Calibration", ["known-null FDR", "known-structure F1"]),
             ("Specificity", ["100 type- and distance-", "matched candidate graphs"]),
             ("Direction", ["W↔H, H↔C, W↔C under", "full conditioning"]),
             ("Regimes", ["state interactions, 24 h prior", "and Niño3.4 states, hemispheres"]),
             ("Timing & transport", ["lag windows, aligned paths,", "vector wind, moisture flux"]),
             ("Independent cloud", ["CERES substitution,", "same-date source check"])]
    tw, gap = 176, 12
    tx0 = (W - (6 * tw + 5 * gap)) / 2
    for k, (title, lines) in enumerate(tiles):
        tx = tx0 + k * (tw + gap)
        e.append(f'<rect x="{tx}" y="{vy + 58}" width="{tw}" height="160" rx="10" fill="white" stroke="{VALID[1]}" stroke-width="1.2"/>')
        e.append(icon(k, tx + tw / 2, vy + 104))
        e.append(t(tx + tw / 2, vy + 154, title, size=13, weight="bold"))
        for j, line in enumerate(lines):
            e.append(t(tx + tw / 2, vy + 176 + 17 * j, line, size=10.8, color=MUTED))
    e.append("</svg>")
    return "\n".join(e)


def icon(k, cx, cy):
    if k == 0:  # calibration: bell curve with threshold
        path = " ".join(f"{'M' if i == 0 else 'L'}{cx - 50 + i * 2:.1f},{cy + 22 - 40 * 2.718 ** (-((i - 25) / 9) ** 2):.1f}" for i in range(51))
        return (f'<path d="{path}" fill="#dab3b6" fill-opacity="0.6" stroke="#b0616a" stroke-width="1.4"/>'
                f'<line x1="{cx + 26}" y1="{cy - 24}" x2="{cx + 26}" y2="{cy + 24}" stroke="#555" stroke-width="1.2" stroke-dasharray="4 3"/>')
    if k == 1:  # specificity: histogram plus observed line
        bars = "".join(f'<rect x="{cx - 46 + 12 * i}" y="{cy + 22 - h}" width="10" height="{h}" fill="#c9ced6"/>' for i, h in enumerate((8, 18, 30, 38, 26, 14, 6)))
        return bars + f'<line x1="{cx + 46}" y1="{cy - 24}" x2="{cx + 46}" y2="{cy + 22}" stroke="#b0616a" stroke-width="2"/>'
    if k == 2:  # direction: two nodes, two arrows
        return (node(cx - 34, cy, "H", 15) + node(cx + 34, cy, "C", 15)
                + f'<path d="M{cx - 16},{cy - 8} Q{cx},{cy - 22} {cx + 16},{cy - 8}" fill="none" stroke="#6f6f6f" stroke-width="2.2" marker-end="url(#arr)"/>'
                + f'<path d="M{cx + 16},{cy + 8} Q{cx},{cy + 22} {cx - 16},{cy + 8}" fill="none" stroke="#b0616a" stroke-width="1.2" stroke-dasharray="4 3" marker-end="url(#arr)"/>')
    if k == 3:  # regimes: two hemispheres
        return (f'<circle cx="{cx}" cy="{cy}" r="24" fill="#f6f4f1" stroke="#9a9a9a" stroke-width="1.2"/>'
                f'<path d="M{cx - 24},{cy} A24,24 0 0 1 {cx + 24},{cy} Z" fill="{VAR["W"][0]}" stroke="none"/>'
                f'<path d="M{cx - 24},{cy} A24,24 0 0 0 {cx + 24},{cy} Z" fill="{VAR["T"][0]}" stroke="none"/>'
                f'<line x1="{cx - 24}" y1="{cy}" x2="{cx + 24}" y2="{cy}" stroke="#777" stroke-width="1"/>'
                + t(cx, cy - 6, "NH", size=9.5) + t(cx, cy + 15, "SH", size=9.5))
    if k == 4:  # timing: lag bars and flux arrow
        bars = "".join(f'<rect x="{cx - 46 + 9 * i}" y="{cy + 20 - h}" width="7" height="{h}" fill="{VAR["H"][1]}"/>' for i, h in enumerate((30, 22, 14, 10, 8, 8, 9, 7, 8, 7, 6)))
        return bars + f'<line x1="{cx - 48}" y1="{cy - 22}" x2="{cx + 44}" y2="{cy - 22}" stroke="{VAR["W"][1]}" stroke-width="2.2" marker-end="url(#arr)"/>'
    return (f'<ellipse cx="{cx - 8}" cy="{cy + 4}" rx="30" ry="15" fill="{VAR["C"][0]}" stroke="{VAR["C"][1]}" stroke-width="1.2"/>'
            f'<ellipse cx="{cx + 14}" cy="{cy - 6}" rx="22" ry="13" fill="{VAR["C"][0]}" stroke="{VAR["C"][1]}" stroke-width="1.2"/>'
            f'<rect x="{cx + 26}" y="{cy - 30}" width="20" height="10" fill="#9a9a9a"/><line x1="{cx + 36}" y1="{cy - 20}" x2="{cx + 22}" y2="{cy - 10}" stroke="#777" stroke-width="1" stroke-dasharray="3 2"/>')


def render(svg: str, name: str):
    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / f"{name}.svg").write_text(svg, encoding="utf-8")
    html = ('<!doctype html><html><head><meta charset="utf-8"><style>@page{size:%dpx %dpx;margin:0}html,body{margin:0;padding:0}svg{display:block}</style>'
            '</head><body>%s</body></html>') % (W, H, svg)
    page = FIG / f"{name}.html"
    page.write_text(html, encoding="utf-8")
    url = page.resolve().as_uri()
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", "--no-pdf-header-footer",
                        f"--print-to-pdf={FIG / (name + '.pdf')}", url], check=True, capture_output=True, timeout=120)
    with tempfile.TemporaryDirectory() as profile:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", "--force-device-scale-factor=2",
                        f"--window-size={W},{H}", "--hide-scrollbars", f"--screenshot={FIG / (name + '.png')}", url], check=True, capture_output=True, timeout=120)
    page.unlink()


if __name__ == "__main__":
    render(build(), "fig01_framework")
    print("rendered", FIG / "fig01_framework.pdf")
