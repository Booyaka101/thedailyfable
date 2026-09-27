"""Compare synthesised rain with real recordings: third-octave long-term spectra (normalised
to their own 1-4 kHz energy so microphones and levels drop out), spectral centroid, crest
factor at 20 ms, and spectrograms side by side.

    python judge.py out.png a.wav b.ogg ...
"""
import sys, subprocess, io
import numpy as np, soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

FS = 48000


def load(path, max_s=60.0):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-t", str(max_s), "-ac", "1", "-ar", str(FS), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def third_octave(y):
    centres = 1000 * 2 ** (np.arange(-15, 14) / 3)      # 31 Hz .. 20 kHz
    f, t, S = spectrogram(y, FS, nperseg=4096, noverlap=2048, scaling="spectrum", mode="psd")
    P = S.mean(1)
    out = []
    for c in centres:
        lo, hi = c / 2 ** (1 / 6), c * 2 ** (1 / 6)
        out.append(P[(f >= lo) & (f < hi)].sum())
    out = np.array(out) + 1e-20
    ref = out[(centres >= 1000) & (centres <= 4000)].sum()
    return centres, 10 * np.log10(out / ref)


def stats(y):
    f, t, S = spectrogram(y, FS, nperseg=2048, noverlap=1024, scaling="spectrum", mode="psd")
    P = S.mean(1)
    cent = (f * P).sum() / P.sum()
    frames = y[: len(y) // 960 * 960].reshape(-1, 960)
    rms = np.sqrt((frames ** 2).mean(1) + 1e-12)
    crest = 20 * np.log10(np.abs(frames).max(1) / rms).mean()
    return cent, crest, 20 * np.log10(rms.std() / rms.mean() + 1e-9)


def main(out, paths):
    n = len(paths)
    fig = plt.figure(figsize=(16, 3.2 * n + 4), facecolor="#111")
    gs = fig.add_gridspec(n + 1, 2, width_ratios=[2.2, 1], height_ratios=[1] * n + [1.6])
    axl = fig.add_subplot(gs[n, :])
    for i, p in enumerate(paths):
        y = load(p)
        name = p.split("/")[-1].split("\\")[-1]
        c, l = third_octave(y)
        cent, crest, mod = stats(y)
        ax = fig.add_subplot(gs[i, 0])
        f, t, S = spectrogram(y[: FS * 12], FS, nperseg=1024, noverlap=768)
        ax.pcolormesh(t, f / 1000, 10 * np.log10(S + 1e-14), cmap="magma", vmin=-140, vmax=-60, shading="auto")
        ax.set_ylim(0, 22); ax.set_ylabel("kHz", color="w"); ax.tick_params(colors="w")
        ax.set_title(f"{name}   centroid {cent:.0f} Hz   crest(20ms) {crest:.1f} dB   level-mod {mod:.1f} dB", color="w", loc="left", fontsize=10)
        ax2 = fig.add_subplot(gs[i, 1])
        f2, t2, S2 = spectrogram(y[FS * 2: FS * 2 + FS // 2], FS, nperseg=256, noverlap=224)
        ax2.pcolormesh(t2 * 1000, f2 / 1000, 10 * np.log10(S2 + 1e-14), cmap="magma", vmin=-140, vmax=-60, shading="auto")
        ax2.set_ylim(0, 22); ax2.tick_params(colors="w"); ax2.set_title("half a second, 5 ms windows", color="w", fontsize=9)
        axl.semilogx(c, l, label=name, lw=2 if "test" in name or "rain_" in name else 1)
        print(f"{name:24s} centroid {cent:6.0f} Hz  crest {crest:5.1f} dB  levelmod {mod:5.1f} dB  "
              f"LTAS@250 {l[c.searchsorted(250)]:6.1f} @1k {l[c.searchsorted(1000)]:6.1f} @4k {l[c.searchsorted(4000)]:6.1f} @8k {l[c.searchsorted(8000)]:6.1f} @16k {l[c.searchsorted(16000)]:6.1f}")
    axl.set_xlabel("Hz", color="w"); axl.set_ylabel("dB re 1-4 kHz band", color="w"); axl.tick_params(colors="w")
    axl.grid(alpha=0.3); axl.legend(fontsize=8); axl.set_xlim(60, 20000); axl.set_ylim(-45, 5)
    for a in fig.axes:
        a.set_facecolor("#1a1a1a")
    fig.tight_layout()
    fig.savefig(out, dpi=100, facecolor="#111")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
