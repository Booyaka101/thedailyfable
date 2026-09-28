"""Have the trained arm write a text onto a page.

    python write.py --policy ../out/policy/final.pt --text ../letter.txt --out ../out/letter

The text is laid out line by line. The page sits on the desk so that the current line's baseline is at a
comfortable writing height; between lines the pen lifts and the page slides up one line spacing, as a
writer pushes the paper away. The rollout is one continuous run of the physics with noise and fatigue,
so the hand tires as the page goes on. Output: trace.npz with the pen tip in desk and page coordinates,
pressure, joints, muscle forces and the page offset at every physics step.
"""
import argparse, os, json, time
import numpy as np, torch
import arm as A, control as C, glyphs as G

RATE = 250
EX = 0.0032                  # x-height, m
LINE = 0.0098                # line spacing, m
MARGIN_L, MARGIN_T = 0.024, 0.030
PAGE_X = 0.030               # desk x of the page's left edge
WRITE_Y = 0.425              # desk y of the current baseline
SLIDE_T = 0.7                # seconds to push the page up a line


def wrap(text, max_w):
    """Greedy line wrap by advance width in x-heights. Blank line = paragraph break."""
    lines = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
            continue
        cur, w = [], 0.0
        for word in para.split():
            ww = G.text_width(word)
            if cur and w + G.SPACE + ww > max_w:
                lines.append(" ".join(cur)); cur, w = [word], ww
            else:
                w += (G.SPACE if cur else 0) + ww; cur.append(word)
        if cur:
            lines.append(" ".join(cur))
    return lines


def build_target(lines, K, rng):
    """Desk-frame target (T,3), page offset per sample (T,2), and line index per sample."""
    segs, offs, lidx = [], [], []
    y_page = A4_H - MARGIN_T - 2 * EX          # first baseline, page coords from the bottom
    prev_end = None
    for li, line in enumerate(lines):
        off = np.array([PAGE_X, WRITE_Y - y_page])          # desk = page + off
        if not line.strip():
            y_page -= 0.5 * LINE                 # paragraph gap is half a line
            continue
        strokes = G.layout(line, EX, MARGIN_L, y_page)
        t, xy, p = G.trajectory(strokes, RATE, K=K * rng.uniform(0.95, 1.05), dwell=0.0 if prev_end is not None else 0.4)
        xy = xy + off
        if prev_end is not None:
            # slide the page: pen up, arm drifts back to the left margin while the paper moves
            n = int(SLIDE_T * RATE)
            s = np.linspace(0, 1, n, endpoint=False)
            ease = 0.5 - 0.5 * np.cos(np.pi * s)
            move = prev_end[None] + (xy[0] - prev_end)[None] * ease[:, None]
            segs.append(np.concatenate([move, np.zeros((n, 1))], 1))
            offs.append(prev_off[None] + (off - prev_off)[None] * ease[:, None])
            lidx.append(np.full(n, li))
        segs.append(np.concatenate([xy, p[:, None]], 1)); offs.append(np.repeat(off[None], len(xy), 0)); lidx.append(np.full(len(xy), li))
        prev_end, prev_off = xy[-1], off
        y_page -= LINE
    # rest at the end
    n = int(1.0 * RATE)
    segs.append(np.concatenate([np.repeat(prev_end[None], n, 0), np.zeros((n, 1))], 1)); offs.append(np.repeat(prev_off[None], n, 0)); lidx.append(np.full(n, li))
    return np.concatenate(segs).astype(np.float32), np.concatenate(offs).astype(np.float32), np.concatenate(lidx)


A4_H = 0.297


def run(policy, target, seed=0, chunk=250 * 20, fatigue=True):
    """Roll the whole target through the arm in chunks (no gradient). Returns traces at the physics rate."""
    g = torch.Generator(device=A.dev).manual_seed(seed)
    arm = A.Arm(1, noise=True, fatigue=fatigue, gen=g)
    tgt = torch.tensor(target, device=A.dev)[None]
    T = tgt.shape[1] // C.TICK * C.TICK
    keys = ("tip", "p", "F", "joints", "a", "fat", "u")
    out = {k: [] for k in keys}
    sensor = None
    with torch.no_grad():
        for t in range(0, T, chunk):
            o = C.rollout(policy, arm, tgt[:, t:min(t + chunk, T)], record=True, trunc=0, sensor=sensor)
            sensor = o["sensor"]
            for k in keys:
                out[k].append(o[k][0])
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=63)
    ap.add_argument("--no-fatigue", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.RandomState(a.seed)
    text = open(a.text, encoding="utf-8").read().rstrip("\n")
    max_w = (0.210 - MARGIN_L - 0.020) / EX
    lines = wrap(text, max_w)
    print(f"{len(lines)} lines, longest {max(len(l) for l in lines)} chars")
    target, offs, lidx = build_target(lines, a.K, rng)
    print(f"target {target.shape[0] / RATE:.1f} s")
    pol = C.Policy().to(A.dev)
    pol.load_state_dict(torch.load(a.policy, map_location=A.dev)["policy"])
    t0 = time.time()
    tr = run(pol, target, seed=a.seed, fatigue=not a.no_fatigue)
    n = len(tr["tip"])
    print(f"rolled {n / RATE:.1f} s in {time.time() - t0:.0f} s")
    err = np.linalg.norm(tr["tip"] - target[:n, :2], axis=1) * 1000
    down = target[:n, 2] > 0.1
    print(f"tracking rms {np.sqrt((err[down] ** 2).mean()):.3f} mm (pen down), max {err[down].max():.2f} mm; fatigue end {tr['fat'].max():.3f}")
    np.savez_compressed(os.path.join(a.out, "trace.npz"), target=target[:n], offset=offs[:n], line=lidx[:n], rate=RATE, **tr)
    json.dump({"lines": lines, "K": a.K, "seed": a.seed, "ex": EX, "line": LINE, "margin_l": MARGIN_L, "margin_t": MARGIN_T,
               "page_x": PAGE_X, "write_y": WRITE_Y}, open(os.path.join(a.out, "layout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
