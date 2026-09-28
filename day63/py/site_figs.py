"""Figures for the day page: the practice strip and the learning curve.

    python site_figs.py ../out/practice ../out/policy/log.txt ../site_figs
"""
import json, os, re, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import ink


def strip(practice, out, attempts=(0, 1, 2, 5, 10, 20, 30)):
    s = json.load(open(os.path.join(practice, "summary.json")))
    rows = []
    for e in s["attempts"]:
        if e["iter"] not in attempts:
            continue
        d = np.load(os.path.join(practice, e["file"]))
        page = ink.Page(size=(0.10, 0.026))
        xy = d["tip"] - d["offset"]
        xy = xy - xy.min(0) + np.array([0.006, 0.006], np.float32)
        page.add(xy, d["p"], 1.0 / float(d["rate"]))
        im = Image.fromarray(page.render_u8(out_size=(page.H // 3, page.W // 3)))
        rows.append((e["iter"], e["rms_mm"], im))
    w, h = rows[0][2].size
    lab = 260
    sheet = Image.new("RGB", (w + lab, h * len(rows)), (246, 243, 236))
    dr = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/georgia.ttf", 34)
    small = ImageFont.truetype("C:/Windows/Fonts/georgia.ttf", 26)
    for i, (it, rms, im) in enumerate(rows):
        sheet.paste(im, (lab, i * h))
        dr.text((24, i * h + h // 2 - 36), f"attempt {it + 1}", fill=(40, 40, 50), font=font)
        dr.text((24, i * h + h // 2 + 6), f"{rms:.2f} mm rms", fill=(120, 120, 130), font=small)
    sheet.save(out, quality=88)
    print("wrote", out, sheet.size)


def curve(log, out):
    ev = [(int(a), float(b)) for a, b in re.findall(r"eval it\s+(\d+) rms ([\d.]+) mm", open(log).read())]
    tr = [(int(a), float(b)) for a, b in re.findall(r"^it\s+(\d+) loss [\d.]+ rms ([\d.]+) mm", open(log).read(), re.M)]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    ax.plot([a for a, _ in tr], [b for _, b in tr], ".", ms=3, color="#9fb0c8", alpha=0.5, label="training batch")
    ax.plot([a for a, _ in ev], [b for _, b in ev], "o-", color="#c0392b", ms=4, label="held-out drills (fixed noise)")
    ax.axvline(30, color="#f0b35b", lw=1, ls="--")
    ax.text(31, 0.63, "checkpoint used for the letter", color="#f0b35b", fontsize=9)
    ax.set_xlabel("practice iteration (one iteration = 96 drills of 1.5 s)")
    ax.set_ylabel("tracking error, mm rms")
    ax.set_ylim(0.45, 0.85)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(0, 320)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out, ev[:3], ev[-1])


if __name__ == "__main__":
    os.makedirs(sys.argv[3], exist_ok=True)
    strip(sys.argv[1], os.path.join(sys.argv[3], "practice.jpg"))
    curve(sys.argv[2], os.path.join(sys.argv[3], "curve.png"))
