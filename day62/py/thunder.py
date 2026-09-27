"""Thunder from a lightning channel's geometry (after Few 1969 and Ribner & Roy 1982).

The channel is a tortuous walk from the ground to the cloud base with a few branches and a
long, nearly horizontal intracloud section. Every segment is a short cylindrical shock: an
N-wave whose amplitude falls as 1/d and radiates strongest broadside to the segment. The
listener hears the sum, each segment arriving after its own distance over c, then the
whole thing passes through air absorption for its distance, a scattering low-pass, and a
ground reflection. The clap is the part of the channel seen broadside; the rumble is the
rest arriving late.

    python thunder.py out.wav          full-length stereo track with every strike in physics.STRIKES
"""
import sys
import numpy as np, soundfile as sf
import physics as P

FS = 48000


def channel(rng, dist_m, az, cloud_base):
    """Segments as (start xyz, end xyz, weight). Listener at origin, y ahead, z up."""
    segs = []
    p = np.array([dist_m * np.sin(az), dist_m * np.cos(az), 0.0])
    u = np.array([0.0, 0.0, 1.0])
    main = []
    while p[2] < cloud_base:
        l = rng.uniform(8, 25)
        q = p + u * l
        segs.append((p.copy(), q.copy(), 1.0))
        main.append(q.copy())
        # mesotortuosity ~16 degrees rms per step, pulled back toward vertical
        u = u + rng.normal(0, 0.28, 3)
        u[2] = abs(u[2]) + 0.25
        u /= np.linalg.norm(u)
        p = q
    # branches off the lower channel, heading down and out
    for _ in range(rng.integers(2, 5)):
        k = rng.integers(0, max(1, len(main) // 2))
        b = main[k].copy()
        v = rng.normal(0, 1, 3); v[2] = -abs(v[2]) - 0.5; v /= np.linalg.norm(v)
        for _ in range(rng.integers(15, 40)):
            l = rng.uniform(8, 25)
            q = b + v * l
            if q[2] < 0:
                break
            segs.append((b.copy(), q.copy(), 0.35))
            v = v + rng.normal(0, 0.3, 3); v /= np.linalg.norm(v)
            b = q
    # intracloud section: 2 to 5 km, nearly horizontal, wandering
    top = main[-1].copy()
    v = rng.normal(0, 1, 3); v[2] = rng.uniform(-0.1, 0.25); v /= np.linalg.norm(v)
    L = rng.uniform(2000, 5000)
    run = 0.0
    b = top
    while run < L:
        l = rng.uniform(10, 30)
        q = b + v * l
        segs.append((b.copy(), q.copy(), 0.5))
        v = v + rng.normal(0, 0.25, 3); v[2] *= 0.5; v /= np.linalg.norm(v)
        b = q
        run += l
    return segs


def nwave(T_s):
    n = max(int(T_s * FS), 4)
    t = np.arange(n) / n
    w = 1 - 2 * t
    # soften the shock fronts slightly: the real front has finite rise
    r = max(2, n // 12)
    w[:r] *= np.linspace(0, 1, r)
    w[-r:] *= np.linspace(1, 0, r)
    return w


def strike_track(tf, dist_km, az_deg, seed, length_s=60.0):
    rng = np.random.default_rng(seed)
    dist = dist_km * 1000
    az = np.radians(az_deg)
    segs = channel(rng, dist, az, rng.uniform(1500, 2500))
    n = int(length_s * FS)
    L, R = np.zeros(n), np.zeros(n)
    for (a, b, wgt) in segs:
        m = (a + b) / 2
        u = (b - a); l = np.linalg.norm(u); u /= l
        d = np.linalg.norm(m)
        rdir = m / d
        broadside = np.sqrt(max(1e-3, 1 - np.dot(u, rdir) ** 2))
        amp = wgt * l / 15.0 * broadside * (1000.0 / d) * rng.uniform(0.6, 1.4)
        T = rng.uniform(0.004, 0.012)
        w = nwave(T) * amp
        t_arr = d / P.C_AIR
        i0 = int(t_arr * FS)
        if i0 + len(w) >= n:
            continue
        saz = np.arctan2(m[0], m[1])
        pnn = np.sin(saz)
        gl, gr = np.sqrt(0.5 * (1 - 0.5 * pnn)), np.sqrt(0.5 * (1 + 0.5 * pnn))
        itd = int(round((0.0875 / P.C_AIR) * (abs(np.arcsin(pnn)) + abs(pnn)) * FS))
        dl, dr = (itd, 0) if pnn > 0 else (0, itd)
        L[i0 + dl:i0 + dl + len(w)] += w * gl
        R[i0 + dr:i0 + dr + len(w)] += w * gr
    # propagation: absorption over the mean distance, scattering low-pass, ground reflection
    f = np.fft.rfftfreq(n, 1 / FS)
    att = 10 ** (-P.air_absorption_db_per_m(f) * dist / 20)
    fc = 1500.0 * (1000.0 / dist) ** 0.7
    lp = 1 / (1 + (f / fc) ** 2)
    hp = (f / 12.0) ** 2 / (1 + (f / 12.0) ** 2)      # nothing below ~12 Hz survives playback anyway
    # scattering by turbulence and terrain smears every pulse over tens of ms, more with distance
    tau_s = 0.03 * (dist / 1000.0) ** 0.5
    ks = rng.standard_normal(int(6 * tau_s * FS)) * np.exp(-np.arange(int(6 * tau_s * FS)) / (tau_s * FS))
    ks /= np.sqrt((ks ** 2).sum())
    KS = np.fft.rfft(ks, n)
    out = []
    for ch in (L, R):
        X = np.fft.rfft(ch) * att * lp * hp * KS
        y = np.fft.irfft(X, n)
        dly = int(0.0026 * FS)
        y[dly:] += 0.6 * y[:-dly]
        out.append(y)
    y = np.stack(out, 1)
    return y


def first_arrival(tf, dist_km, az_deg, seed):
    """Seconds after the flash until the nearest bit of channel is heard (same channel as the track)."""
    rng = np.random.default_rng(seed)
    segs = channel(rng, dist_km * 1000, np.radians(az_deg), rng.uniform(1500, 2500))
    dmin = min(np.linalg.norm((a + b) / 2) for (a, b, w) in segs)
    return dmin / P.C_AIR


def arrivals():
    return {str(tf): first_arrival(tf, dist, az, 100 + k) for k, (tf, dist, az) in enumerate(P.STRIKES)}


def render_all(duration=P.DURATION):
    n = int(duration * FS)
    track = np.zeros((n, 2), np.float32)
    for k, (tf, dist, az) in enumerate(P.STRIKES):
        y = strike_track(tf, dist, az, seed=100 + k)
        # level: a strike at 1.4 km peaks near full scale; others by 1/d, absorption already in
        pk = np.abs(y).max()
        y = y / pk * 0.95 * min(1.0, 1.4 / dist) ** 0.8
        i0 = int(tf * FS)
        m = min(len(y), n - i0)
        track[i0:i0 + m] += y[:m]
        t_first = tf + dist * 1000 / P.C_AIR
        print(f"strike {k} t={tf} d={dist} km az={az}  thunder starts {t_first:.1f}s  peak {np.abs(y).max():.2f}", flush=True)
    return track


if __name__ == "__main__":
    y = render_all()
    sf.write(sys.argv[1], y, FS, subtype="FLOAT")
