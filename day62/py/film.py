"""Mix the rain and the thunder into one soundtrack and print what the meter says.

    python film.py ../out/mix.wav
"""
import sys, subprocess
import numpy as np, soundfile as sf

RAIN_GAIN = 0.85
THUNDER_GAIN = 1.0


def main(out):
    rain, fs = sf.read("../out/rain.wav", dtype="float32")
    thunder, fs2 = sf.read("../out/thunder.wav", dtype="float32")
    assert fs == fs2
    n = min(len(rain), len(thunder))
    y = rain[:n] * RAIN_GAIN + thunder[:n] * THUNDER_GAIN
    # rain alone briefly touches 0 dBFS at the peak of the storm: a soft knee above 0.8 keeps
    # every sample under full scale without a compressor pumping the storm
    a = np.abs(y)
    knee = 0.8
    over = a > knee
    y[over] = np.sign(y[over]) * (knee + (1 - knee) * np.tanh((a[over] - knee) / (1 - knee)))
    # 4 s fade in from silence, 8 s fade out
    y[: fs * 4] *= np.linspace(0, 1, fs * 4)[:, None] ** 2
    y[-fs * 8:] *= np.linspace(1, 0, fs * 8)[:, None] ** 2
    y *= 0.9
    sf.write(out, y, fs, subtype="PCM_24")
    print(f"peak {np.abs(y).max():.3f}  rain peak {np.abs(rain).max():.3f}  thunder peak {np.abs(thunder).max():.3f}")
    r = subprocess.run(["ffmpeg", "-v", "info", "-i", out, "-af", "ebur128=peak=true", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    print(r[r.index("Summary"):].strip())


if __name__ == "__main__":
    main(sys.argv[1])
