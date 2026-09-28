"""A cursive skeleton alphabet, and the machinery that turns text into a pen-tip target trajectory.

Glyph coordinates are in x-heights: baseline y = 0, x-height y = 1, ascenders reach 2, descenders -1.
A lowercase glyph's first stroke runs from its entry (where the join from the previous letter arrives)
to its exit (where the join to the next letter leaves). Later strokes are deferred: dots and crosses
are added after the word is finished, the way a hand does it. Capitals, digits and punctuation stand
alone: the pen lifts after them.

Timing follows the two-thirds power law of handwriting (Lacquaniti, Terzuolo & Viviani 1983): pen speed
falls where curvature is high, v = K * kappa^(-1/3), bounded above and below.
"""
import math
import numpy as np

# (advance width, high exit?, strokes). Strokes are point lists; the first is the connected one.
LOWER = {
    "a": (1.35, False, [[(0, .45), (.55, .9), (.85, 1.0), (.45, .88), (.12, .5), (.25, .08), (.55, 0), (.85, .3), (.93, .95), (.98, .5), (1.05, .05), (1.2, 0), (1.35, .4)]]),
    "b": (1.15, True, [[(0, .35), (.4, 1.3), (.52, 2.0), (.44, 1.95), (.34, 1.2), (.3, .05), (.5, 0), (.82, .3), (.8, .68), (.62, .82), (.9, .85), (1.15, .92)]]),
    "c": (1.15, False, [[(0, .45), (.55, .9), (.9, 1.0), (.55, .92), (.2, .55), (.25, .1), (.6, 0), (.9, .18), (1.15, .4)]]),
    "d": (1.4, False, [[(0, .45), (.55, .9), (.85, 1.0), (.45, .88), (.12, .5), (.25, .08), (.55, 0), (.85, .3), (.95, 1.2), (1.02, 2.0), (1.0, 1.3), (1.05, .05), (1.2, 0), (1.4, .4)]]),
    "e": (1.05, False, [[(0, .4), (.45, .7), (.75, .95), (.55, 1.0), (.2, .7), (.15, .3), (.35, .02), (.65, 0), (.9, .2), (1.05, .4)]]),
    "f": (1.05, False, [[(0, .4), (.4, 1.3), (.5, 2.0), (.42, 1.95), (.35, 1.0), (.4, -.4), (.42, -.95), (.55, -.9), (.6, -.5), (.5, .2), (.8, .35), (1.05, .4)], [(.15, .95), (.75, .95)]]),
    "g": (1.35, False, [[(0, .45), (.55, .9), (.85, 1.0), (.45, .88), (.12, .5), (.25, .08), (.55, 0), (.85, .3), (.93, .95), (.98, .2), (.92, -.65), (.7, -1.0), (.5, -.85), (.6, -.4), (1.0, .1), (1.35, .4)]]),
    "h": (1.4, False, [[(0, .35), (.4, 1.3), (.5, 2.0), (.43, 1.95), (.35, 1.2), (.32, .05), (.4, .6), (.7, 1.0), (.95, .85), (1.05, .4), (1.08, .05), (1.2, 0), (1.4, .4)]]),
    "i": (.75, False, [[(0, .45), (.3, .8), (.42, 1.0), (.4, .5), (.42, .05), (.55, 0), (.75, .4)], [(.44, 1.4)]]),
    "j": (.8, False, [[(0, .45), (.3, .8), (.45, 1.0), (.5, .2), (.42, -.65), (.22, -1.0), (.05, -.85), (.15, -.45), (.55, .15), (.8, .4)], [(.48, 1.4)]]),
    "k": (1.3, False, [[(0, .35), (.4, 1.3), (.5, 2.0), (.43, 1.95), (.35, 1.2), (.32, .05), (.4, .6), (.75, 1.0), (.95, .9), (.75, .6), (.5, .5), (.8, .4), (1.0, .1), (1.15, 0), (1.3, .4)]]),
    "l": (.9, False, [[(0, .35), (.35, 1.3), (.45, 2.0), (.38, 1.95), (.3, 1.2), (.3, .05), (.5, 0), (.7, .1), (.9, .4)]]),
    "m": (1.95, False, [[(0, .45), (.25, .8), (.4, 1.0), (.4, .4), (.42, .05), (.5, .5), (.7, .95), (.92, .95), (1.0, .5), (1.02, .05), (1.1, .5), (1.32, .95), (1.55, .95), (1.62, .5), (1.65, .05), (1.78, 0), (1.95, .4)]]),
    "n": (1.35, False, [[(0, .45), (.25, .8), (.4, 1.0), (.4, .4), (.42, .05), (.5, .5), (.72, .95), (.95, .95), (1.02, .5), (1.05, .05), (1.18, 0), (1.35, .4)]]),
    "o": (1.1, True, [[(0, .45), (.5, .9), (.75, 1.0), (.4, .85), (.12, .5), (.25, .08), (.55, 0), (.85, .3), (.9, .75), (.72, .95), (.9, .9), (1.1, .92)]]),
    "p": (1.3, False, [[(0, .45), (.3, .8), (.45, 1.0), (.5, .2), (.48, -.6), (.45, -1.0), (.48, -.3), (.55, .55), (.78, .95), (1.0, .85), (1.05, .45), (.85, .05), (.55, 0), (.9, .05), (1.3, .4)]]),
    "q": (1.35, False, [[(0, .45), (.55, .9), (.85, 1.0), (.45, .88), (.12, .5), (.25, .08), (.55, 0), (.85, .3), (.93, .95), (.98, .2), (1.0, -.65), (1.05, -1.0), (1.2, -.75), (1.05, -.2), (1.1, .15), (1.35, .4)]]),
    "r": (1.05, False, [[(0, .45), (.35, .85), (.45, 1.0), (.5, .72), (.68, .95), (.92, 1.0), (.86, .78), (.8, .4), (.85, .05), (1.05, .4)]]),
    "s": (1.0, False, [[(0, .45), (.35, .8), (.6, 1.0), (.45, .8), (.25, .55), (.55, .35), (.75, .15), (.6, 0), (.3, .05), (.4, .12), (.75, .25), (1.0, .4)]]),
    "t": (1.0, False, [[(0, .4), (.35, 1.0), (.45, 1.6), (.42, 1.3), (.4, .05), (.55, 0), (.78, .1), (1.0, .4)], [(.1, 1.0), (.85, 1.0)]]),
    "u": (1.35, False, [[(0, .45), (.3, .8), (.42, 1.0), (.4, .4), (.45, .05), (.65, 0), (.9, .3), (1.0, 1.0), (1.0, .5), (1.05, .05), (1.18, 0), (1.35, .4)]]),
    "v": (1.15, True, [[(0, .45), (.3, .8), (.42, 1.0), (.42, .5), (.5, .02), (.7, .1), (.95, .55), (1.0, .95), (.9, 1.0), (1.15, .92)]]),
    "w": (1.7, True, [[(0, .45), (.3, .8), (.42, 1.0), (.42, .5), (.5, .02), (.7, .1), (.9, .5), (.95, 1.0), (1.0, .5), (1.05, .02), (1.25, .1), (1.5, .55), (1.55, .95), (1.45, 1.0), (1.7, .92)]]),
    "x": (1.2, False, [[(0, .45), (.3, .8), (.5, 1.0), (.7, .6), (.85, .15), (.95, 0), (1.2, .4)], [(1.0, 1.0), (.75, .55), (.45, .15), (.2, 0)]]),
    "y": (1.35, False, [[(0, .45), (.3, .8), (.42, 1.0), (.4, .4), (.45, .05), (.65, 0), (.9, .3), (1.0, 1.0), (1.05, .2), (.95, -.65), (.72, -1.0), (.55, -.85), (.65, -.4), (1.05, .15), (1.35, .4)]]),
    "z": (1.25, False, [[(0, .45), (.3, .8), (.45, 1.0), (.9, 1.0), (.55, .55), (.35, .3), (.7, .28), (.8, 0), (.6, -.5), (.4, -.9), (.2, -.85), (.3, -.5), (.75, .0), (1.0, .15), (1.25, .4)]]),
}

# standalone glyphs: (advance, strokes). No entry/exit joins.
UPPER = {
    "A": (1.6, [[(.05, 0), (.5, 1.2), (.85, 2.0), (1.15, 1.2), (1.5, 0)], [(.35, .8), (1.2, .8)]]),
    "B": (1.4, [[(.25, 2.0), (.2, 0)], [(.2, 2.0), (.9, 2.0), (1.15, 1.7), (1.05, 1.2), (.75, 1.05), (.2, 1.05), (.85, 1.05), (1.25, .8), (1.3, .3), (1.0, 0), (.2, 0)]]),
    "C": (1.5, [[(1.35, 1.7), (1.0, 2.0), (.5, 1.9), (.15, 1.3), (.15, .6), (.5, .05), (1.0, 0), (1.4, .35)]]),
    "D": (1.5, [[(.25, 2.0), (.2, 0)], [(.2, 2.0), (.8, 2.0), (1.2, 1.7), (1.35, 1.0), (1.2, .3), (.8, 0), (.2, 0)]]),
    "E": (1.35, [[(1.2, 2.0), (.25, 2.0), (.2, 0), (1.2, 0)], [(.2, 1.05), (.95, 1.05)]]),
    "F": (1.3, [[(1.2, 2.0), (.25, 2.0), (.2, 0)], [(.2, 1.05), (.9, 1.05)]]),
    "G": (1.6, [[(1.4, 1.7), (1.05, 2.0), (.5, 1.9), (.15, 1.3), (.15, .6), (.5, .05), (1.05, 0), (1.4, .3), (1.45, .9), (.9, .9)]]),
    "H": (1.6, [[(.2, 2.0), (.2, 0)], [(.2, 1.05), (1.35, 1.05)], [(1.35, 2.0), (1.35, 0)]]),
    "I": (.7, [[(.3, 2.0), (.3, 0)]]),
    "J": (1.1, [[(.9, 2.0), (.9, .6), (.7, .05), (.35, 0), (.1, .3)]]),
    "K": (1.5, [[(.2, 2.0), (.2, 0)], [(1.3, 2.0), (.6, 1.1), (.2, .85)], [(.6, 1.1), (1.35, 0)]]),
    "L": (1.3, [[(.25, 2.0), (.2, 0), (1.2, 0)]]),
    "M": (1.9, [[(.15, 0), (.25, 2.0), (.9, .3), (1.55, 2.0), (1.65, 0)]]),
    "N": (1.6, [[(.15, 0), (.25, 2.0), (1.35, 0), (1.4, 2.0)]]),
    "O": (1.6, [[(.8, 2.0), (.3, 1.7), (.15, 1.0), (.3, .3), (.8, 0), (1.3, .3), (1.45, 1.0), (1.3, 1.7), (.8, 2.0), (.5, 1.85)]]),
    "P": (1.4, [[(.25, 0), (.25, 2.0), (.9, 2.0), (1.25, 1.7), (1.25, 1.2), (.9, .95), (.25, .95)]]),
    "Q": (1.6, [[(.8, 2.0), (.3, 1.7), (.15, 1.0), (.3, .3), (.8, 0), (1.3, .3), (1.45, 1.0), (1.3, 1.7), (.8, 2.0), (.5, 1.85)], [(1.0, .5), (1.5, -.15)]]),
    "R": (1.5, [[(.25, 0), (.25, 2.0), (.9, 2.0), (1.25, 1.7), (1.25, 1.2), (.9, .95), (.25, .95), (.85, .9), (1.4, 0)]]),
    "S": (1.3, [[(1.15, 1.7), (.85, 2.0), (.4, 1.9), (.2, 1.5), (.45, 1.1), (.9, .9), (1.15, .5), (.95, .05), (.5, 0), (.1, .3)]]),
    "T": (1.4, [[(.1, 2.0), (1.3, 2.0)], [(.7, 2.0), (.7, 0)]]),
    "U": (1.6, [[(.2, 2.0), (.2, .6), (.5, .05), (.9, 0), (1.25, .3), (1.35, .8), (1.35, 2.0)]]),
    "V": (1.6, [[(.1, 2.0), (.75, 0), (1.45, 2.0)]]),
    "W": (2.2, [[(.1, 2.0), (.55, 0), (1.05, 1.6), (1.55, 0), (2.05, 2.0)]]),
    "X": (1.5, [[(.1, 2.0), (1.35, 0)], [(1.35, 2.0), (.1, 0)]]),
    "Y": (1.5, [[(.1, 2.0), (.75, .95), (1.4, 2.0)], [(.75, .95), (.75, 0)]]),
    "Z": (1.4, [[(.15, 2.0), (1.25, 2.0), (.15, 0), (1.3, 0)]]),
}

DIGITS = {
    "0": (1.2, [[(.55, 2.0), (.2, 1.5), (.15, 1.0), (.2, .4), (.55, 0), (.9, .4), (.95, 1.0), (.9, 1.5), (.55, 2.0)]]),
    "1": (.8, [[(.15, 1.6), (.5, 2.0), (.5, 0)]]),
    "2": (1.2, [[(.15, 1.6), (.5, 2.0), (.95, 1.75), (.9, 1.3), (.5, .8), (.15, .3), (.1, 0), (1.0, 0)]]),
    "3": (1.2, [[(.15, 1.8), (.5, 2.0), (.9, 1.7), (.8, 1.25), (.5, 1.05), (.85, .95), (1.0, .5), (.8, .05), (.4, 0), (.1, .25)]]),
    "4": (1.2, [[(.8, 0), (.8, 2.0), (.1, .6), (1.05, .6)]]),
    "5": (1.2, [[(.95, 2.0), (.25, 2.0), (.15, 1.1), (.55, 1.25), (.95, 1.0), (1.0, .5), (.75, .05), (.4, 0), (.1, .25)]]),
    "6": (1.2, [[(.9, 1.9), (.5, 1.6), (.2, 1.0), (.15, .5), (.4, .05), (.8, .05), (1.0, .5), (.85, .95), (.5, 1.05), (.2, .7)]]),
    "7": (1.2, [[(.1, 2.0), (1.0, 2.0), (.55, 1.0), (.35, 0)]]),
    "8": (1.2, [[(.55, 1.05), (.25, 1.4), (.35, 1.9), (.7, 2.0), (.95, 1.6), (.7, 1.2), (.4, 1.0), (.15, .55), (.35, .05), (.75, 0), (1.0, .5), (.8, .95), (.55, 1.05)]]),
    "9": (1.2, [[(.3, .1), (.7, .4), (.95, 1.0), (1.0, 1.5), (.75, 1.95), (.4, 1.95), (.15, 1.5), (.3, 1.05), (.65, .95), (1.0, 1.3)]]),
}

PUNCT = {
    ".": (.5, [[(.2, 0), (.24, .02)]]),
    ",": (.5, [[(.22, .05), (.25, -.15), (.1, -.45)]]),
    "'": (.4, [[(.22, 2.0), (.15, 1.55)]]),
    "-": (.9, [[(.15, .55), (.75, .55)]]),
    ":": (.5, [[(.2, 1.0), (.24, 1.02)], [(.2, 0), (.24, .02)]]),
    ";": (.5, [[(.2, 1.0), (.24, 1.02)], [(.22, .05), (.25, -.15), (.1, -.45)]]),
    "?": (1.1, [[(.15, 1.6), (.4, 2.0), (.85, 1.85), (.9, 1.4), (.55, 1.0), (.5, .6)], [(.5, 0), (.54, .02)]]),
    "!": (.6, [[(.3, 2.0), (.3, .6)], [(.3, 0), (.34, .02)]]),
    "(": (.7, [[(.55, 2.1), (.2, 1.4), (.15, .6), (.5, -.3)]]),
    ")": (.7, [[(.15, 2.1), (.5, 1.4), (.55, .6), (.2, -.3)]]),
}

SPACE = 0.9
SLANT = 0.22        # x shear per unit y, about 12 degrees


def catmull_rom(pts, n_per=12, alpha=0.5):
    """Centripetal Catmull-Rom through pts (N,2) -> dense polyline."""
    P = np.asarray(pts, float)
    if len(P) == 1:
        return P.copy()
    if len(P) == 2:
        t = np.linspace(0, 1, n_per + 1)[:, None]
        return P[0] * (1 - t) + P[1] * t
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        d = lambda a, b: max(np.linalg.norm(b - a) ** alpha, 1e-6)
        t0, t1, t2, t3 = 0, d(p0, p1), 0, 0
        t2 = t1 + d(p1, p2)
        t3 = t2 + d(p2, p3)
        t = np.linspace(t1, t2, n_per, endpoint=False)[:, None]
        A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
        out.append(C)
    out.append(P[-2][None])
    return np.vstack(out)


def layout(text, ex, x0, y0):
    """Text (one line) -> list of pen-down strokes, each an (N,2) array in metres, in writing order.

    x0, y0: page position of the line's left baseline start. Returns strokes in order of execution:
    the connected body of each word, then that word's deferred dots and crosses.
    """
    strokes = []
    x = 0.0
    body, deferred = [], []
    prev_high = False
    in_word = False

    def flush():
        nonlocal body, deferred
        if body:
            strokes.append(np.array(body))
        for d in deferred:
            strokes.append(np.array(d))
        body, deferred = [], []

    def place(pts, xoff):
        return [(xoff + px + SLANT * py, py) for px, py in pts]

    for ch in text:
        if ch == " ":
            flush()
            x += SPACE
            in_word = False
            prev_high = False
            continue
        if ch in LOWER:
            w, high, strs = LOWER[ch]
            main = strs[0]
            if in_word and prev_high:
                main = main[1:]           # coming from a high exit: skip the baseline entry point
            elif not in_word:
                main = main[1:] if main[0][1] < 0.6 and len(main) > 2 and False else main
            body += place(main, x)
            for s in strs[1:]:
                deferred.append(place(s, x))
            in_word, prev_high = True, high
            x += w
        else:
            table = UPPER if ch in UPPER else DIGITS if ch in DIGITS else PUNCT if ch in PUNCT else None
            if table is None:
                continue
            flush()
            w, strs = table[ch]
            for s in strs:
                strokes.append(np.array(place(s, x)))
            x += w + 0.15
            in_word, prev_high = False, False
    flush()
    out = []
    for s in strokes:
        s = s * ex
        s[:, 0] += x0
        s[:, 1] += y0
        out.append(s)
    return out


def text_width(text):
    """Advance width of a string in x-heights, as layout() would place it."""
    w = 0.0
    for ch in text:
        if ch == " ":
            w += SPACE
        elif ch in LOWER:
            w += LOWER[ch][0]
        else:
            table = UPPER if ch in UPPER else DIGITS if ch in DIGITS else PUNCT if ch in PUNCT else None
            if table is not None:
                w += table[ch][0] + 0.15
    return w


def densify(stroke, step=0.00015):
    """Spline through the skeleton points, resampled to roughly uniform arc length."""
    if len(stroke) == 1:
        return stroke.copy()
    c = catmull_rom(stroke, n_per=16)
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] < step:
        return c[[0, -1]]
    n = max(2, int(s[-1] / step))
    ss = np.linspace(0, s[-1], n)
    return np.stack([np.interp(ss, s, c[:, 0]), np.interp(ss, s, c[:, 1])], 1)


def curvature(c):
    d1 = np.gradient(c, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5 + 1e-12
    return num / den


def trajectory(strokes, rate, K=0.055, vmin=0.02, vmax=0.16, lift_speed=0.35, lift_height_t=0.08, dwell=0.06):
    """Strokes -> (t, xy, pressure) sampled at `rate` Hz, following the two-thirds power law.

    K sets the tempo: v = K * kappa^(-1/3) in m/s with kappa in 1/m. Pen-up moves between strokes travel
    at lift_speed with pressure 0; the pen rests `dwell` seconds before the first stroke.
    """
    T, XY, P = [], [], []
    t = 0.0
    dt = 1.0 / rate

    def add_move(a, b, speed, pressure_from, pressure_to):
        nonlocal t
        d = np.linalg.norm(b - a)
        dur = max(d / speed, lift_height_t)
        n = max(2, int(round(dur * rate)))
        s = np.linspace(0, 1, n, endpoint=False)
        ease = 0.5 - 0.5 * np.cos(np.pi * s)
        xy = a[None] + (b - a)[None] * ease[:, None]
        pr = np.where(s < 0.25, pressure_from * (1 - s / 0.25), 0.0)
        pr = np.where(s > 0.75, pressure_to * (s - 0.75) / 0.25, pr)
        T.append(t + s * dur); XY.append(xy); P.append(pr)
        t += dur

    prev = None
    for st in strokes:
        c = densify(st)
        if prev is not None:
            add_move(prev, c[0], lift_speed, 0.0, 0.0)
        elif dwell > 0:
            n = int(dwell * rate)
            T.append(t + np.arange(n) * dt); XY.append(np.repeat(c[:1], n, 0)); P.append(np.zeros(n))
            t += n * dt
        if len(c) < 2:
            # a dot: press and release in place
            n = int(0.09 * rate)
            s = np.linspace(0, 1, n)
            T.append(t + s * 0.09); XY.append(np.repeat(c[:1], n, 0)); P.append(np.sin(np.pi * s) ** 0.5)
            t += 0.09
            prev = c[0]
            continue
        kap = curvature(c)
        # smooth the curvature so the speed profile is not jittery
        m = min(9, len(kap))
        k = np.convolve(kap, np.ones(m) / m, mode="same")
        v = np.clip(K * np.maximum(k, 1.0) ** (-1 / 3), vmin, vmax)
        # slow into and out of each stroke
        seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        ramp = np.clip(s / 0.0015, 0.25, 1) * np.clip((s[-1] - s) / 0.0015, 0.25, 1)
        v = v * ramp
        tt = np.concatenate([[0], np.cumsum(seg / (0.5 * (v[1:] + v[:-1])))])
        n = max(2, int(tt[-1] * rate))
        ts = np.linspace(0, tt[-1], n)
        xy = np.stack([np.interp(ts, tt, c[:, 0]), np.interp(ts, tt, c[:, 1])], 1)
        # a little more pressure on the downstrokes, as a nib does
        dy = np.gradient(xy[:, 1])
        dl = np.linalg.norm(np.gradient(xy, axis=0), axis=1) + 1e-9
        pr = 0.62 + 0.25 * np.clip(-dy / dl, 0, 1)
        T.append(t + ts); XY.append(xy); P.append(pr)
        t += tt[-1]
        prev = c[-1]
    T = np.concatenate(T); XY = np.concatenate(XY); P = np.concatenate(P)
    # resample onto the uniform clock
    tu = np.arange(0, T[-1], dt)
    XYu = np.stack([np.interp(tu, T, XY[:, 0]), np.interp(tu, T, XY[:, 1])], 1)
    Pu = np.interp(tu, T, P)
    return tu, XYu, Pu


def preview(path="../scratch/glyphs.png"):
    from PIL import Image, ImageDraw
    ex = 0.003
    lines = ["abcdefghijklm", "nopqrstuvwxyz", "ABCDEFGHIJKLM", "NOPQRSTUVWXYZ", "0123456789 .,':;?!()-",
             "the quick brown fox jumps over the lazy dog",
             "a hand that learned to write, day sixty-three",
             "owner write brown ferry rare error arm"]
    scale = 22000  # px per metre
    W, H = 3600, 2400
    im = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(im)
    for li, line in enumerate(lines):
        y0 = 0.020 + li * 0.012
        strokes = layout(line, ex, 0.005, y0)
        for st in strokes:
            c = densify(st)
            pts = [(x * scale, H - y * scale) for x, y in c]
            if len(pts) == 1:
                x, y = pts[0]; dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=0)
            else:
                dr.line(pts, fill=0, width=4)
        yb = H - y0 * scale
        dr.line([(0, yb), (W, yb)], fill=225, width=1)
        dr.line([(0, yb - ex * scale), (W, yb - ex * scale)], fill=235, width=1)
    im.save(path)
    print("saved", path)


if __name__ == "__main__":
    import os
    os.makedirs("../scratch", exist_ok=True)
    preview()
    t, xy, p = trajectory(layout("the quick brown fox", 0.003, 0, 0), 500)
    print(f"'the quick brown fox' takes {t[-1]:.2f} s, {len(t)} samples, mean speed {np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()/t[-1]*100:.1f} cm/s")
