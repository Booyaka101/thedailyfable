"""The picture: a pond in the rain, from the same drops the sound is made of.

Surface: linear capillary-gravity waves, omega^2 = g k + (sigma/rho) k^3, advanced exactly
in Fourier space on a 2048 x 2048 grid at 1 mm per cell, viscous damping 2 nu k^2 plus a
surface-film term. Each landing drop dents the surface with a Gaussian crater scaled to its
size. Camera 0.7 m above the water pitched 50 degrees down; every pixel reflects an overcast
sky by Fresnel and shows dark water otherwise. Lightning flashes light the sky from the
strike's azimuth at the flash time; the thunder in the soundtrack arrives d/c later.

    python pond.py out.mp4 [--start s] [--dur s] [--fps 60] [--w 2560 --h 1440] [--preview]
"""
import argparse, subprocess, sys, time, os
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import physics as P
import thunder

ARRIVAL = thunder.arrivals()

dev = torch.device("cuda")
N = 2048
DX = 1e-3
G, SIGMA, RHO, NU = 9.81, 0.072, 1000.0, 1.0e-6
NU_EFF = 1.5e-6
GAMMA0 = 0.6
CAM_POS = np.array([0.0, -0.45, 0.7])
PITCH = np.radians(50.0)
VFOV = np.radians(36.0)
DOMAIN_CENTER = np.array([0.0, 0.25])
FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_B = "C:/Windows/Fonts/segoeuil.ttf"


class Surface:
    def __init__(self):
        kx = torch.fft.fftfreq(N, DX, device=dev) * 2 * np.pi
        ky = torch.fft.rfftfreq(N, DX, device=dev) * 2 * np.pi
        k = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
        self.omega = torch.sqrt(G * k + (SIGMA / RHO) * k ** 3)
        self.gamma = 2 * NU_EFF * k ** 2 + GAMMA0
        self.H = torch.zeros((N, N // 2 + 1), dtype=torch.complex64, device=dev)
        self.V = torch.zeros_like(self.H)
        self.h = torch.zeros((N, N), device=dev)
        self.k = k

    def dent(self, xs, ys, Ds):
        """xs, ys in metres (domain coords), Ds in mm. Adds craters to the height field."""
        if len(xs) == 0:
            return
        field = torch.zeros((N, N), device=dev)
        ix = torch.as_tensor((xs / DX + N / 2), device=dev, dtype=torch.float32)
        iy = torch.as_tensor((ys / DX + N / 2), device=dev, dtype=torch.float32)
        D = torch.as_tensor(Ds, device=dev, dtype=torch.float32)
        sig = torch.clamp(2.0 * D, max=9.0)   # crater radius in cells (1 cell = 1 mm)
        depth = 2.5e-3 * D     # metres
        R = 24
        off = torch.arange(-R, R + 1, device=dev, dtype=torch.float32)
        ox, oy = off[None, :, None], off[None, None, :]
        cx, cy = torch.round(ix), torch.round(iy)
        gx, gy = (cx[:, None, None] + ox - ix[:, None, None]), (cy[:, None, None] + oy - iy[:, None, None])
        g = torch.exp(-(gx ** 2 + gy ** 2) / (2 * sig[:, None, None] ** 2))
        # a crater is a hole with a raised rim: mexican-hat so the mean displacement is ~0
        g = g * (1 - (gx ** 2 + gy ** 2) / (2 * sig[:, None, None] ** 2))
        stamp = -depth[:, None, None] * g
        xi = ((cx[:, None, None] + ox).long() % N)
        yi = ((cy[:, None, None] + oy).long() % N)
        idx = (xi * N + yi).reshape(-1)
        field.view(-1).index_add_(0, idx, stamp.reshape(-1))
        self.H += torch.fft.rfft2(field)

    def step(self, dt):
        w = self.omega
        c, s = torch.cos(w * dt), torch.sin(w * dt)
        safe = torch.where(w > 0, w, torch.ones_like(w))
        Hn = self.H * c + torch.where(w > 0, self.V / safe * s, torch.zeros_like(self.V))
        Vn = -self.H * w * s + self.V * c
        decay = torch.exp(-self.gamma * dt)
        self.H, self.V = Hn * decay, Vn * decay
        self.H[0, 0] = 0
        self.h = torch.fft.irfft2(self.H, s=(N, N))


class Camera:
    def __init__(self, W, Hh, vfov=VFOV):
        self.W, self.Hh = W, Hh
        f = 1.0 / np.tan(vfov / 2)
        ys, xs = torch.meshgrid(torch.arange(Hh, device=dev), torch.arange(W, device=dev), indexing="ij")
        u = (xs + 0.5) / W * 2 - 1
        vv = 1 - (ys + 0.5) / Hh * 2
        aspect = W / Hh
        d = torch.stack([u * aspect / f, torch.ones_like(u), vv / f], -1)     # camera space: x right, y fwd, z up
        d = d / d.norm(dim=-1, keepdim=True)
        cp, sp = np.cos(PITCH), np.sin(PITCH)
        # pitch down about x
        dy = d[..., 1] * cp + d[..., 2] * sp
        dz = -d[..., 1] * sp + d[..., 2] * cp
        d = torch.stack([d[..., 0], dy, dz], -1)
        self.dir = d
        t = -CAM_POS[2] / d[..., 2].clamp(max=-1e-4)
        hit = torch.tensor(CAM_POS, device=dev, dtype=torch.float32)[None, None] + d * t[..., None]
        self.hit = hit
        gx = (hit[..., 0] - DOMAIN_CENTER[0]) / (N * DX / 2)
        gy = (hit[..., 1] - DOMAIN_CENTER[1]) / (N * DX / 2)
        self.grid = torch.stack([gy, gx], -1)[None]        # grid_sample wants (x=cols, y=rows): cols index = y axis of field
        self.dist = t
        # world azimuth of the hit point from the listener (for HUD nothing, for flash directionality)
        self.inside = (gx.abs() < 1) & (gy.abs() < 1)

    def normals(self, h):
        hx = (torch.roll(h, -1, 0) - torch.roll(h, 1, 0)) / (2 * DX)
        hy = (torch.roll(h, -1, 1) - torch.roll(h, 1, 1)) / (2 * DX)
        nm = torch.stack([-hx, -hy], 0)[None]                    # (1,2,N,N)
        n2 = torch.nn.functional.grid_sample(nm, self.grid, mode="bilinear", padding_mode="border", align_corners=False)[0]
        n = torch.stack([n2[0], n2[1], torch.ones_like(n2[0])], -1)
        return n / n.norm(dim=-1, keepdim=True)


def sky(r, flash, flash_az):
    """Overcast luminance for reflected directions r (...,3). Bright toward the horizon and
    toward a thin patch ahead-left; lightning adds a lobe from its azimuth."""
    el = r[..., 2].clamp(0, 1)
    az = torch.atan2(r[..., 0], r[..., 1])
    base = 0.18 + 1.0 * (1 - el) ** 2.5
    patch = 0.35 * torch.exp(-((az + 0.5) ** 2) / 0.5 - ((el - 0.25) ** 2) / 0.04)
    L = base + patch
    if flash > 0:
        lobe = 0.35 + 0.65 * torch.exp(-((torch.remainder(az - flash_az + np.pi, 2 * np.pi) - np.pi) ** 2) / 0.6)
        L = L * (1 + flash * lobe * (0.6 + 0.8 * (1 - el)))
    tint = torch.tensor([0.86, 0.90, 0.97], device=dev)
    return L[..., None] * tint


def flash_env(t):
    """Lightning brightness at time t from the strike list: a few return strokes each."""
    val, az = 0.0, 0.0
    for (tf, dist, azd) in P.STRIKES:
        dt = t - tf
        if 0 <= dt < 1.0:
            rng = np.random.default_rng(int(tf))
            nstroke = rng.integers(2, 5)
            for s in range(nstroke):
                ts = s * rng.uniform(0.04, 0.09)
                if dt >= ts:
                    val += (1.0 if s == 0 else rng.uniform(0.3, 0.7)) * np.exp(-(dt - ts) / rng.uniform(0.02, 0.05))
            val *= 12.0 / max(dist, 1.0) ** 0.5
            az = np.radians(azd)
    return val, az


def make_hud(text, W, size=36):
    lines = text.count(chr(10)) + 1
    img = Image.new("RGBA", (W - 80, int(size * 1.35) * lines + 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f1 = ImageFont.truetype(FONT, size)
    d.multiline_text((0, 0), text, font=f1, fill=(255, 255, 255, 170), spacing=int(size * 0.25))
    a = np.asarray(img).astype(np.float32) / 255
    return torch.as_tensor(a, device=dev)


def card(W, Hh, lines, sizes, y0):
    img = Image.new("RGBA", (W, Hh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    y = y0
    for txt, sz in zip(lines, sizes):
        f = ImageFont.truetype(FONT_B if sz > 60 else FONT, sz)
        w = d.textlength(txt, font=f)
        d.text(((W - w) / 2, y), txt, font=f, fill=(255, 255, 255, 235))
        y += sz * 1.5
    a = np.asarray(img).astype(np.float32) / 255
    return torch.as_tensor(a, device=dev)


def composite(frame, overlay, alpha, x0=0, y0=0):
    h, w = overlay.shape[:2]
    reg = frame[y0:y0 + h, x0:x0 + w]
    a = overlay[..., 3:4] * alpha
    frame[y0:y0 + h, x0:x0 + w] = reg * (1 - a) + overlay[..., :3] * a


def run(out, start, dur, fps, W, Hh, preview, seed=11):
    portrait = Hh > W
    vfov = np.radians(56.0) if portrait else VFOV
    surf, cam = Surface(), Camera(W, Hh, vfov)
    dt = 1.0 / fps
    nframes = int(round(dur * fps))
    water = torch.tensor([0.030, 0.045, 0.040], device=dev)
    cos_i = (-cam.dir[..., 2]).clamp(1e-3, 1)
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-r", str(fps),
           "-i", "-", "-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-b:v", "60M", "-maxrate", "90M",
           "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    chunk_dt = 0.5
    ev_cache = {}
    hud_t, hud_img = -1, None
    title = card(W, Hh, ["RAIN", "twenty minutes of a storm, every drop computed"], [150, 40], Hh * 0.36)
    endc = card(W, Hh, ["Every drop was drawn from the Marshall-Palmer size distribution at the rain rate of the moment.",
                        "Each one's splash, the crater it opened, and the bubble it trapped were synthesised from their physics",
                        "and placed around a listener 1.5 m above the water. The pond is the wave equation on 2048 x 2048 cells.",
                        "Thunder is the sum of N-waves from a tortuous lightning channel, arriving at the speed of sound.",
                        "the daily fable, day 62"], [30, 30, 30, 30, 34], Hh * 0.34)
    t0w = time.time()
    half = N * DX / 2
    cam_cp, cam_sp, cam_f = np.cos(PITCH), np.sin(PITCH), 1.0 / np.tan(vfov / 2)
    for fi in range(nframes):
        t = start + fi * dt
        ci = int(t // chunk_dt)
        for c in (ci, ci + 1):
            if c not in ev_cache:
                ev_cache[c] = P.gen_chunk_events(c, chunk_dt, seed=seed)
        for c in [c for c in ev_cache if c < ci]:
            del ev_cache[c]
        ev = ev_cache[ci]
        tt = ev["t"] + ci * chunk_dt
        m = (tt >= t) & (tt < t + dt) & ev["near"]
        xs, ys = ev["x"][m] - DOMAIN_CENTER[0], ev["y"][m] - DOMAIN_CENTER[1]
        m2 = (np.abs(xs) < half) & (np.abs(ys) < half)
        surf.dent(xs[m2], ys[m2], ev["D"][m][m2])
        surf.step(dt)
        n = cam.normals(surf.h)
        d = cam.dir
        cos_t = (-(d * n).sum(-1)).clamp(1e-3, 1)
        r = d + 2 * cos_t[..., None] * n
        F = 0.02 + 0.98 * (1 - cos_t) ** 5
        fl, faz = flash_env(t)
        col = F[..., None] * sky(r, fl, faz) + (1 - F[..., None]) * water * (1 + 0.5 * fl)
        # drops in the air: every drop that lands in the next 0.16 s is on its way down at its
        # terminal speed; draw the bit of its fall covered in this frame as a faint streak
        streak = torch.zeros((Hh, W), device=dev)
        for c in (ci, ci + 1):
            e2 = ev_cache[c]
            th = e2["t"] + c * chunk_dt
            mm = (th >= t) & (th < t + 0.16) & e2["near"] & (np.abs(e2["x"]) < 1.1) & (e2["y"] > -0.35) & (e2["y"] < 1.4)
            if mm.sum() == 0:
                continue
            vt = torch.as_tensor(P.vterm(e2["D"][mm]), device=dev, dtype=torch.float32)
            thm = torch.as_tensor(th[mm] - t, device=dev, dtype=torch.float32)
            z0 = vt * thm
            z1 = (vt * (thm - dt)).clamp(min=0)
            ns = 192
            frac = torch.linspace(0, 1, ns, device=dev)
            zz = z0[:, None] + (z1 - z0)[:, None] * frac[None, :]
            px = torch.as_tensor(e2["x"][mm] - CAM_POS[0], device=dev, dtype=torch.float32)[:, None].expand(-1, ns)
            py = torch.as_tensor(e2["y"][mm] - CAM_POS[1], device=dev, dtype=torch.float32)[:, None].expand(-1, ns)
            pz = zz - CAM_POS[2]
            ycam = py * cam_cp - pz * cam_sp
            zcam = py * cam_sp + pz * cam_cp
            ok = ycam > 0.05
            yc = ycam.clamp(min=0.05)
            u = (px / yc) * cam_f / (W / Hh)
            vv = (zcam / yc) * cam_f
            xp = ((u + 1) / 2 * W).long()
            yp = ((1 - vv) / 2 * Hh).long()
            ok &= (xp >= 0) & (xp < W) & (yp >= 0) & (yp < Hh)
            dist2 = px ** 2 + ycam ** 2 + zcam ** 2
            Dm = torch.as_tensor(e2["D"][mm], device=dev, dtype=torch.float32)[:, None].expand(-1, ns)
            bright = 0.03 * Dm / dist2.clamp(min=0.05) / ns * 24
            idx = (yp * W + xp)[ok]
            streak.view(-1).index_add_(0, idx, bright[ok])
        col = col + streak.clamp(0, 0.5)[..., None] * torch.tensor([0.85, 0.9, 1.0], device=dev)
        # tone map
        col = col / (1 + col)
        col = col ** (1 / 2.2)
        # HUD, once a second
        if int(t) != hud_t:
            hud_t = int(t)
            R = ev["R"]
            per_s = (len(ev["t"]) - (~ev["near"]).sum() + ev["n_far_true"]) / chunk_dt
            cnt = f"{per_s / 1e6:.1f} million" if per_s >= 1e6 else f"{per_s / 1e3:.0f} thousand"
            if portrait:
                txt = f"{t // 60:02.0f}:{t % 60:02.0f}     {R:.1f} mm/h" + "\n" + f"{cnt} drops a second within 30 m"
            else:
                txt = f"{t // 60:02.0f}:{t % 60:02.0f}     {R:.1f} mm/h     {cnt} drops a second on the water within 30 m"
            for (tf, dist, azd) in P.STRIKES:
                ta = ARRIVAL[str(tf)]
                if tf <= t < tf + ta:
                    txt += "\n" + f"lightning {dist:.1f} km away, thunder in {tf + ta - t:.0f} s"
            hud_img = make_hud(txt, W, 44 if portrait else 36)
        composite(col, hud_img, 1.0, 40, Hh - hud_img.shape[0] - (300 if portrait else 10))
        if t < 8:
            a = 1.0 if t < 5 else (8 - t) / 3
            col = col * (0.15 + 0.85 * (1 - a)) if t < 8 else col
            composite(col, title, a)
        if t > P.DURATION - 14:
            a = min(1.0, (t - (P.DURATION - 14)) / 4)
            col = col * (1 - 0.85 * a)
            composite(col, endc, a)
        if t > P.DURATION - 3:
            col = col * (P.DURATION - t) / 3
        frame = (col.clamp(0, 1) * 255 + torch.rand_like(col) * 0.999).to(torch.uint8).cpu().numpy()
        proc.stdin.write(frame.tobytes())
        if preview and (fi % fps == 0 or (fl > 2.0 and fi % 3 == 0)):
            Image.fromarray(frame).save(f"../scratch/pond_{t:08.2f}.png")
        if fi % (fps * 10) == 0:
            el = time.time() - t0w
            print(f"t={t:7.1f}s  frame {fi}/{nframes}  {fi / max(el, 1e-3):5.1f} fps  drops this frame {int(m2.sum())}  h rms {surf.h.std().item() * 1000:.2f} mm", flush=True)
    proc.stdin.close()
    proc.wait()
    print("done", out, time.time() - t0w)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--dur", type=float, default=P.DURATION)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--w", type=int, default=2560)
    ap.add_argument("--h", type=int, default=1440)
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    run(a.out, a.start, a.dur, a.fps, a.w, a.h, a.preview)
