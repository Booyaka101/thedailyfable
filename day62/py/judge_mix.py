"""Full-length judgement of the mix: long-term spectrogram, 10 s loudness track, and the
thunder onset for each strike measured from the track against tf + d/c.

    python judge_mix.py ../out/mix.wav ../scratch/mix_judge.png
"""
import sys
import numpy as np, soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import spectrogram, butter, sosfiltfilt
import physics as P


def main(path, out):
    y, fs = sf.read(path, dtype="float32")
    m = y.mean(1)
    f, t, S = spectrogram(m, fs, nperseg=8192, noverlap=0)
    S = 10 * np.log10(S + 1e-16)
    # 10 s rms track
    fr = m[: len(m) // (fs * 10) * fs * 10].reshape(-1, fs * 10)
    rms = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    # thunder: energy below 150 Hz, 50 ms windows
    sos = butter(4, 150, "low", fs=fs, output="sos")
    lo = sosfiltfilt(sos, m)
    w = fs // 20
    e = (lo[: len(lo) // w * w].reshape(-1, w) ** 2).mean(1)
    tt = np.arange(len(e)) * w / fs
    print("strike   flash t   expected thunder   measured onset   error")
    for (tf, dist, az) in P.STRIKES:
        exp = tf + dist * 1000 / P.C_AIR
        win = (tt > tf) & (tt < tf + 60)
        base = np.median(e[(tt > tf - 5) & (tt < tf)])
        idx = np.nonzero(win & (e > 8 * base + 1e-10))[0]
        onset = tt[idx[0]] if idx.size else float("nan")
        print(f"{tf:7.0f}  {dist:5.1f} km   {exp:9.1f}   {onset:9.1f}   {onset - exp:+6.2f}")
    fig, ax = plt.subplots(3, 1, figsize=(18, 11), facecolor="#111", gridspec_kw=dict(height_ratios=[3, 1.2, 1.2]))
    ax[0].pcolormesh(t, f / 1000, S, cmap="magma", vmin=-130, vmax=-50, shading="auto")
    ax[0].set_ylim(0, 20); ax[0].set_ylabel("kHz")
    ax[1].plot(np.arange(len(rms)) * 10, rms, color="#8cf"); ax[1].set_ylabel("rms dBFS / 10 s")
    tp = np.linspace(0, P.DURATION, 600)
    ax[1].twinx().semilogy(tp, P.storm_profile(tp), color="#fc8", lw=0.8)
    ax[2].semilogy(tt, e + 1e-12, color="#f88", lw=0.5); ax[2].set_ylabel("<150 Hz energy")
    for (tf, dist, az) in P.STRIKES:
        ax[2].axvline(tf, color="w", lw=0.5, alpha=0.5)
        ax[2].axvline(tf + dist * 1000 / P.C_AIR, color="#8f8", lw=0.5, alpha=0.7)
    for a in ax:
        a.set_facecolor("#1a1a1a"); a.tick_params(colors="w"); a.yaxis.label.set_color("w"); a.set_xlim(0, P.DURATION)
    fig.tight_layout(); fig.savefig(out, dpi=90, facecolor="#111")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
