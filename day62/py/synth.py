"""Airborne sound of rain on open water, drop by drop, on the GPU.

Each drop makes a splash transient (a short shaped-noise grain keyed to its size and its
distance, the far ones dulled by air absorption) and, if it traps a bubble, a Minnaert
tone that decays at the bubble's own damping and chirps up a few percent as it rises.
Sources are placed around a listener 1.5 m above the water; level falls as 1/d and each
ear gets its own delay and gain from the azimuth.

    python synth.py out.wav [--start s] [--dur s] [--rate mm/h]
"""
import argparse, sys, time
import numpy as np, torch, soundfile as sf
import physics as P

FS = 48000
DT = 0.5
CHUNK = int(FS * DT)
L_IMP = 256
L_BUB = 2048
PAD = max(L_BUB, 1024) + 64 + int(0.06 * FS)
SIZE_EDGES = np.geomspace(P.D_MIN, P.D_MAX, 9)          # 8 size classes
DIST_CLASSES = np.array([1.5, 3.0, 6.0, 12.0, 20.0, 30.0])
N_VAR = 8
K_BUB = 2.5
K_CR = 3.0
L_CR = 1024
K_MASTER = 0.02
dev = torch.device("cuda")


def build_grain_bank(seed=3):
    """(8 sizes, 6 distances, 8 variants, L_IMP) splash grains, unit peak before absorption."""
    rng = np.random.default_rng(seed)
    t = np.arange(L_IMP) / FS
    f = np.fft.rfftfreq(L_IMP, 1 / FS)
    bank = np.zeros((8, 6, N_VAR, L_IMP), np.float32)
    for s in range(8):
        D = np.sqrt(SIZE_EDGES[s] * SIZE_EDGES[s + 1])
        fc = 9000.0 * D ** -0.6
        tau = 0.3e-3 * D ** 0.7
        shape = (f / fc) / (1 + (f / fc) ** 2) ** 1.5
        for v in range(N_VAR):
            noise = rng.standard_normal(L_IMP)
            env = (t / 0.1e-3) * np.exp(-t / tau)
            env = np.minimum(env, 1.0) * np.exp(-t / tau)
            g = noise * env
            G = np.fft.rfft(g) * shape
            for q, d in enumerate(DIST_CLASSES):
                att = 10 ** (-P.air_absorption_db_per_m(f) * d / 20)
                gg = np.fft.irfft(G * att, L_IMP)
                gg[-24:] *= np.hanning(48)[-24:]
                bank[s, q, v] = gg
        # normalise by the nearest-distance peak so absorption stays relative
        pk = np.abs(bank[s, 0]).max(axis=-1, keepdims=True)
        bank[s] /= pk[None]
    return torch.tensor(bank, device=dev)


BANK = None


def synth_chunk(i, out_prev_tail, rate=None, seed=11):
    """Returns (stereo chunk (CHUNK,2) float32 numpy, tail to add to the next chunk, stats)."""
    global BANK
    if BANK is None:
        BANK = build_grain_bank()
    ev = P.gen_chunk_events(i, DT, seed=seed, R=rate)
    rng = ev["rng"]
    n = ev["t"].size
    buf = torch.zeros(2, CHUNK + PAD, device=dev)
    if out_prev_tail is not None:
        buf[:, :out_prev_tail.shape[1]] += out_prev_tail
    if n == 0:
        return buf[:, :CHUNK].T.cpu().numpy(), buf[:, CHUNK:].clone(), dict(R=ev["R"], n=0, nb=0)

    D, r, az = ev["D"], ev["r"], ev["az"]
    d = np.sqrt(r ** 2 + P.H_EAR ** 2)
    v = P.vterm(D)
    A = (D / 1.0) * (v / 6.0) ** 1.5 / d * ev["scale"]
    lateral = np.arcsin(np.sin(az))
    p = np.sin(az)
    gL, gR = np.sqrt(0.5 * (1 - 0.7 * p)), np.sqrt(0.5 * (1 + 0.7 * p))
    itd = (0.0875 / P.C_AIR) * (np.abs(lateral) + np.abs(np.sin(lateral)))
    itd_s = np.rint(itd * FS).astype(np.int64)
    dL = np.where(p > 0, itd_s, 0)       # source on the right: left ear hears it later
    dR = np.where(p < 0, itd_s, 0)
    t0 = np.rint(ev["t"] * FS).astype(np.int64)
    s_idx = np.clip(np.searchsorted(SIZE_EDGES, D) - 1, 0, 7)
    q_idx = np.abs(np.log(d)[:, None] - np.log(DIST_CLASSES)[None, :]).argmin(1)
    v_idx = rng.integers(0, N_VAR, n)

    ar = torch.arange(L_IMP, device=dev)
    flat = buf.view(-1)
    W = CHUNK + PAD
    for b0 in range(0, n, 100000):
        sl = slice(b0, b0 + 100000)
        g = BANK[torch.as_tensor(s_idx[sl], device=dev), torch.as_tensor(q_idx[sl], device=dev),
                 torch.as_tensor(v_idx[sl], device=dev)]
        At = torch.as_tensor(A[sl], device=dev, dtype=torch.float32)
        tt = torch.as_tensor(t0[sl], device=dev)
        for ch, gg, dd in ((0, gL, dL), (1, gR, dR)):
            idx = (tt + torch.as_tensor(dd[sl], device=dev))[:, None] + ar[None, :] + ch * W
            sig = g * (At * torch.as_tensor(gg[sl], device=dev, dtype=torch.float32))[:, None]
            flat.index_add_(0, idx.reshape(-1), sig.reshape(-1))

    # crater: the hole a drop opens and the water that closes it displace air; the radiated
    # pressure follows the second derivative of the crater volume. Volume ~ kinetic energy,
    # formation time ~ 1.2 ms per mm of drop, shape sin^4 in time so the onset is smooth.
    ci = np.nonzero(D > 0.7)[0]
    if ci.size:
        Dc = D[ci]
        T = np.minimum(1.2e-3 * Dc, L_CR / FS)
        T3 = 1.2e-3 * 3.0
        Ac = K_CR * (Dc / 3.0) ** 3 * (v[ci] / 7.9) ** 2 * (T3 / T) ** 2 / d[ci] * ev["scale"][ci]
        tvec = torch.arange(L_CR, device=dev, dtype=torch.float32) / FS
        arc = torch.arange(L_CR, device=dev)
        for b0 in range(0, ci.size, 40000):
            sl = slice(b0, b0 + 40000)
            Tt = torch.as_tensor(T[sl], device=dev, dtype=torch.float32)[:, None]
            x = np.pi * tvec[None] / Tt
            s2x = torch.sin(x) ** 2
            shape = s2x * (12 * torch.cos(x) ** 2 - 4 * s2x) / 12.0
            shape = torch.where(tvec[None] < Tt, shape, torch.zeros_like(shape))
            sig = shape * torch.as_tensor(Ac[sl], device=dev, dtype=torch.float32)[:, None]
            tt = torch.as_tensor(t0[ci][sl], device=dev)
            for ch, gg, dd in ((0, gL, dL), (1, gR, dR)):
                idx = (tt + torch.as_tensor(dd[ci][sl], device=dev))[:, None] + arc[None, :] + ch * W
                s3 = sig * torch.as_tensor(gg[ci][sl], device=dev, dtype=torch.float32)[:, None]
                flat.index_add_(0, idx.reshape(-1), s3.reshape(-1))

    # bubbles
    has, Rb, delay = P.bubble_params(D, rng)
    bi = np.nonzero(has)[0]
    nb = bi.size
    if nb:
        f0 = P.MINNAERT / Rb[bi]
        delta = P.bubble_damping(f0)
        tau = 1.0 / (np.pi * f0 * delta)
        att = 10 ** (-P.air_absorption_db_per_m(f0) * d[bi] / 20)
        Ab = K_BUB * (Rb[bi] / 1e-3) * att / d[bi] * ev["scale"][bi]
        tb = t0[bi] + np.rint(delay[bi] * FS).astype(np.int64)
        tvec = torch.arange(L_BUB, device=dev, dtype=torch.float32) / FS
        fade = torch.ones(L_BUB, device=dev)
        fade[-64:] = torch.linspace(1, 0, 64, device=dev)
        for b0 in range(0, nb, 20000):
            sl = slice(b0, b0 + 20000)
            f0t = torch.as_tensor(f0[sl], device=dev, dtype=torch.float32)[:, None]
            taut = torch.as_tensor(tau[sl], device=dev, dtype=torch.float32)[:, None]
            env = torch.exp(-tvec[None] / taut)
            chirp = 0.03 * (tvec[None] - taut * (1 - env))
            sig = torch.sin(2 * np.pi * f0t * (tvec[None] + chirp)) * env
            sig *= torch.clamp(tvec[None] / 0.15e-3, max=1.0) * fade[None]
            sig *= torch.as_tensor(Ab[sl], device=dev, dtype=torch.float32)[:, None]
            tt = torch.as_tensor(tb[sl], device=dev)
            arb = torch.arange(L_BUB, device=dev)
            for ch, gg, dd in ((0, gL, dL), (1, gR, dR)):
                idx = (tt + torch.as_tensor(dd[bi][sl], device=dev))[:, None] + arb[None, :] + ch * W
                s2 = sig * torch.as_tensor(gg[bi][sl], device=dev, dtype=torch.float32)[:, None]
                flat.index_add_(0, idx.reshape(-1), s2.reshape(-1))

    buf *= 1.0
    out = buf[:, :CHUNK].T.contiguous().cpu().numpy()
    tail = buf[:, CHUNK:].clone()
    return out, tail, dict(R=ev["R"], n=n, nb=nb, near=int(ev["near"].sum()), n_far_true=ev["n_far_true"])


def render(start, dur, rate=None, seed=11, log=None):
    i0 = int(round(start / DT))
    nch = int(round(dur / DT))
    out = np.zeros((nch * CHUNK, 2), np.float32)
    tail = None
    t_start = time.time()
    for k in range(nch):
        chunk, tail, st = synth_chunk(i0 + k, tail, rate=rate, seed=seed)
        out[k * CHUNK:(k + 1) * CHUNK] = chunk
        if log is not None and k % 20 == 0:
            print(f"  t={start + k * DT:7.1f}s R={st['R']:5.1f} mm/h drops={st['n']:7d} "
                  f"(near {st.get('near', 0)}, far true {st.get('n_far_true', 0):.0f}) bubbles={st['nb']:6d} "
                  f"peak={np.abs(chunk).max() * K_MASTER:.3f}  [{time.time() - t_start:.0f}s]", file=log, flush=True)
    return out * K_MASTER


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=P.DURATION)
    ap.add_argument("--rate", type=float, default=None, help="fixed rain rate mm/h instead of the storm")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    y = render(a.start, a.dur, a.rate, a.seed, log=sys.stdout)
    pk = np.abs(y).max()
    print(f"peak {pk:.3f}  rms {np.sqrt((y ** 2).mean()):.4f}")
    sf.write(a.out, y, FS, subtype="FLOAT")
