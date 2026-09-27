"""Rain physics shared by the audio and the picture.

Drop sizes follow Marshall & Palmer (1948): N(D) = N0 exp(-Lambda D), N0 = 8000 m^-3 mm^-1,
Lambda = 4.1 R^-0.21 mm^-1 for a rain rate R in mm/h. Terminal fall speed from Atlas,
Srivastava & Sekhon (1973): v = 9.65 - 10.3 exp(-0.6 D) m/s. The flux of drops through a
horizontal surface is N(D) v(D) per m^2 per s per mm of diameter.

Which drops trap a bubble, and how big, follows Pumphrey, Crum & Bjorno (1989) and
Prosperetti & Oguz (1993): 0.8 to 1.1 mm drops at terminal speed trap a small bubble almost
every time (the "regular entrainment" window), mid-size drops rarely do, large drops trap a
larger bubble about half the time as the crater collapses. The bubble rings at the Minnaert
frequency f = 3.26 / R (R in m), which is the drip "plink".
"""
import numpy as np

N0 = 8000.0          # m^-3 mm^-1
D_MIN, D_MAX = 0.3, 6.0   # mm; below 0.3 mm the splash is inaudible, above 6 mm drops break up
C_AIR = 343.0
MINNAERT = 3.26      # Hz m


def lam(R):
    return 4.1 * np.maximum(R, 1e-3) ** -0.21


def vterm(D):
    return np.maximum(9.65 - 10.3 * np.exp(-0.6 * D), 0.05)


def dsd(D, R):
    return N0 * np.exp(-lam(R)[..., None] * D) if np.ndim(R) else N0 * np.exp(-lam(R) * D)


_Dgrid = np.linspace(D_MIN, D_MAX, 2000)


def flux_table(R):
    """Returns (drops per m^2 per s, cdf over _Dgrid) for a rain rate R mm/h."""
    f = dsd(_Dgrid, R) * vterm(_Dgrid)
    cdf = np.cumsum(f)
    total = cdf[-1] * (_Dgrid[1] - _Dgrid[0])
    return total, cdf / cdf[-1]


def rain_rate_check(R):
    """Integrate the volume flux back to mm/h, to check the distribution is self-consistent."""
    f = dsd(_Dgrid, R) * vterm(_Dgrid) * (np.pi / 6) * _Dgrid ** 3
    return np.trapezoid(f, _Dgrid) * 3.6e-3


def sample_D(cdf, n, rng):
    u = rng.random(n)
    return np.interp(u, cdf, _Dgrid)


def bubble_params(D, rng):
    """Per drop: (has_bubble, radius_m, delay_s). Vectorised over D in mm."""
    n = D.size
    p = np.where(D < 0.8, 0.02,
        np.where(D <= 1.1, 0.9 * np.exp(-((D - 0.95) / 0.12) ** 2) + 0.05,
        np.where(D < 2.2, 0.08, 0.5)))
    has = rng.random(n) < p
    # radius in mm, lognormal around a size that tracks the regime
    mu = np.where(D <= 1.1, 0.22, np.where(D < 2.2, 0.45, 0.95 * (D / 3.0)))
    sig = np.where(D <= 1.1, 0.15, 0.3)
    Rb = mu * np.exp(sig * rng.standard_normal(n))
    Rb = np.clip(Rb, 0.12, 3.0)
    # the bubble pinches off as the crater retracts; a few ms for small drops, tens for large
    delay = (0.003 + 0.006 * np.sqrt(D)) * np.exp(0.25 * rng.standard_normal(n))
    return has, Rb * 1e-3, delay


def bubble_damping(f):
    """Total damping constant delta (thermal + radiation + viscous), Ainslie & Leighton 2011 scale.
    Q = 1/delta. ~0.03 at 3 kHz, ~0.07 at 16 kHz."""
    return 0.03 * np.sqrt(f / 3000.0)


def air_absorption_db_per_m(f_hz):
    """ISO 9613-1 style, 20 C, ~70 % RH, fitted through 4 kHz = 0.03 dB/m."""
    return 0.03 * (np.maximum(f_hz, 50.0) / 4000.0) ** 1.66


# ---------------- storm profile ----------------
STORM_KEYS = [  # (seconds, mm/h): a cell drifting over, 20 minutes
    (0, 0.15), (60, 0.5), (150, 1.5), (240, 4), (360, 10), (480, 22), (560, 40), (600, 55),
    (640, 48), (700, 30), (780, 14), (880, 6), (980, 2.5), (1080, 0.8), (1160, 0.3), (1200, 0.1)]
DURATION = 1200.0

# lightning: (flash time s, horizontal distance km, azimuth deg from ahead)
STRIKES = [(205, 9.0, 40), (330, 6.5, 25), (432, 4.2, 15), (522, 2.6, -10), (586, 1.4, -35),
           (627, 2.1, -60), (690, 3.8, -80), (772, 5.5, -95), (905, 8.0, -110), (1052, 11.0, -120)]


def storm_profile(t, seed=7):
    """Rain rate mm/h at times t (s), the keyframe curve times a gusty modulation with a
    ~20 s correlation time. Deterministic for a seed."""
    ts = np.array([k[0] for k in STORM_KEYS], float)
    rs = np.log(np.array([k[1] for k in STORM_KEYS], float))
    base = np.exp(np.interp(t, ts, rs))
    rng = np.random.default_rng(seed)
    grid = np.arange(0, DURATION + 60, 2.0)
    noise = rng.standard_normal(grid.size)
    k = np.exp(-0.5 * (np.arange(-30, 31) / 5.0) ** 2)
    noise = np.convolve(noise, k / np.sqrt((k ** 2).sum()), mode="same")
    mod = np.exp(0.35 * np.interp(t, grid, noise))
    return base * mod


if __name__ == "__main__":
    for R in [0.5, 2, 10, 40]:
        tot, _ = flux_table(R)
        print(f"R={R:5.1f} mm/h  Lambda={lam(R):.2f}  drops/m2/s (D>0.3mm)={tot:8.1f}  "
              f"MP integrates back to {rain_rate_check(R):.2f} mm/h")
    t = np.linspace(0, DURATION, 7)
    print("profile", np.round(storm_profile(t), 2))


# ---------------- event generation (shared by sound and picture) ----------------
H_EAR = 1.5          # listener height above the water, m
R_NEAR = 4.0         # every drop inside this radius is synthesised individually
R_FAR = 30.0         # beyond this the rain is inaudible over the nearer rain
CAP_FAR = 60000      # far-ring events per chunk; the rest is carried by sqrt(N/M) scaling


def gen_chunk_events(i, dt, seed=11, R=None):
    """All drops that land within R_FAR of the listener during chunk i (dt seconds).
    Deterministic per (seed, i). Returns dict of arrays: t (s within chunk), x, y (m), D (mm),
    scale (amplitude factor for far-ring subsampling), near (bool)."""
    rng = np.random.default_rng([seed, i])
    t0 = i * dt
    if R is None:
        R = float(storm_profile(np.array([t0 + dt / 2]))[0])
    flux, cdf = flux_table(R)
    n_near = rng.poisson(flux * np.pi * R_NEAR ** 2 * dt)
    n_far_true = flux * np.pi * (R_FAR ** 2 - R_NEAR ** 2) * dt
    n_far = int(min(rng.poisson(n_far_true), CAP_FAR))
    far_scale = np.sqrt(n_far_true / max(n_far, 1)) if n_far_true > CAP_FAR else 1.0
    n = n_near + n_far
    r = np.empty(n)
    r[:n_near] = R_NEAR * np.sqrt(rng.random(n_near))
    r[n_near:] = np.sqrt(R_NEAR ** 2 + (R_FAR ** 2 - R_NEAR ** 2) * rng.random(n_far))
    az = rng.random(n) * 2 * np.pi
    D = sample_D(cdf, n, rng)
    scale = np.ones(n)
    scale[n_near:] = far_scale
    near = np.zeros(n, bool)
    near[:n_near] = True
    return dict(t=rng.random(n) * dt, x=r * np.sin(az), y=r * np.cos(az), r=r, az=az, D=D,
                scale=scale, near=near, R=R, n_far_true=n_far_true, rng=rng)
