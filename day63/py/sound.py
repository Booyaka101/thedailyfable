"""The sound of the pen, from the trace: nib friction on paper, the tap of landing, the paper sliding up.

    python sound.py ../out/letter/trace.npz ../out/letter/pen.wav

Friction is filtered noise whose level rises with pressure and speed and whose spectrum shifts up with
speed (the nib crosses more fibres per second). Push strokes (away from the writer) scratch harder than
pull strokes. Landings are short clicks whose loudness follows how fast the pressure rose.
"""
import sys
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter, sosfilt

FS = 48000


def band_noise(n, lo, hi, rng, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=FS, output="sos")
    return sosfilt(sos, rng.standard_normal(n)).astype(np.float32)


def resonant_click(fc, decay, dur=0.03):
    t = np.arange(int(dur * FS)) / FS
    return (np.exp(-t / decay) * np.sin(2 * np.pi * fc * t)).astype(np.float32)


def synth(trace, seed=0):
    rng = np.random.default_rng(seed)
    rate = int(trace["rate"])
    tip = trace["tip"]; p = trace["p"]; off = trace["offset"]
    n_ctl = len(tip)
    dur = n_ctl / rate
    n = int(dur * FS)
    t_ctl = np.arange(n_ctl) / rate
    t = np.arange(n) / FS
    v = np.gradient(tip, axis=0) * rate                         # m/s in desk frame
    speed = np.linalg.norm(v, axis=1)
    push = np.clip(v[:, 1] / (speed + 1e-6), 0, 1)
    down = np.clip((p - 0.05) / 0.3, 0, 1)
    # friction level and spectral weights at the control rate
    lvl = down ** 0.8 * np.clip(speed / 0.06, 0, 2.5) ** 0.6 * (1 + 0.6 * push)
    s = np.clip(speed / 0.10, 0, 1)
    w_lo, w_mid, w_hi = (1 - s) * 0.8 + 0.2, 0.7 + 0.3 * s, 0.15 + 0.85 * s ** 1.5
    up = lambda a: np.interp(t, t_ctl, a).astype(np.float32)
    lvl_a = up(lvl)
    # smooth the level so the noise does not buzz at the control rate
    k = int(0.004 * FS); lvl_a = np.convolve(lvl_a, np.ones(k) / k, mode="same")
    out = (band_noise(n, 300, 1200, rng) * up(w_lo) * 0.9 + band_noise(n, 1200, 3500, rng) * up(w_mid)
           + band_noise(n, 3500, 9000, rng) * up(w_hi) * 0.7) * lvl_a
    # paper grain: amplitude modulation from the fibre field under the nib, a few hundred fibres per cm
    grain = 1 + 0.35 * np.interp(t, t_ctl, np.cumsum(speed) / rate)      # position along the stroke, m
    grain = 1 + 0.35 * np.sin(2 * np.pi * 300 * 100 * grain) * np.interp(t, t_ctl, s)
    out = out * grain.astype(np.float32)
    # landings and lifts
    dp = np.gradient(p) * rate
    land = np.where((dp > 1.5) & (np.roll(p, 1) < 0.08))[0]
    lift = np.where((dp < -1.5) & (np.roll(p, -1) < 0.05))[0]
    ck = resonant_click(2400, 0.006) * 0.6 + resonant_click(900, 0.012) * 0.5
    lk = resonant_click(1800, 0.004) * 0.35
    for i in land:
        j = int(i / rate * FS)
        amp = min(dp[i] / 8, 1.5) * rng.uniform(0.7, 1.0)
        out[j:j + len(ck)] += ck[: n - j] * amp * 0.5
    for i in lift:
        j = int(i / rate * FS)
        out[j:j + len(lk)] += lk[: n - j] * rng.uniform(0.5, 1.0) * 0.3
    # the page sliding: where the page offset changes
    doff = np.linalg.norm(np.gradient(off, axis=0), axis=1) * rate
    slide = up(np.clip(doff / 0.02, 0, 1))
    rustle = band_noise(n, 80, 900, rng) * slide * 0.5 + band_noise(n, 900, 4000, rng) * slide * 0.12
    out = out + rustle
    # a small dry room: two early reflections, no tail
    out = out + 0.18 * np.concatenate([np.zeros(int(0.007 * FS), np.float32), out[: n - int(0.007 * FS)]]) \
              + 0.10 * np.concatenate([np.zeros(int(0.013 * FS), np.float32), out[: n - int(0.013 * FS)]])
    # room floor
    out = out + band_noise(n, 40, 400, rng) * 0.004
    out = out / (np.abs(out).max() + 1e-9) * 0.7
    st = np.stack([out * 0.95, out * 1.0 + 0.02 * np.roll(out, 9)], 1)
    return st


if __name__ == "__main__":
    tr = np.load(sys.argv[1])
    y = synth(tr)
    sf.write(sys.argv[2], y, FS, subtype="PCM_24")
    print("wrote", sys.argv[2], f"{len(y) / FS:.1f} s")
