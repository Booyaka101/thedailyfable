"""Ink on paper. A pen trace (page position, pressure, time) becomes deposited ink on a 600 dpi canvas.

Deposition is per unit time, not per unit length, so a slow pen leaves a darker, slightly wider line, and
the nib's footprint grows with pressure. The canvas keeps ink density; `render` turns a crop of it into
an RGB image with paper texture, fibre bleed and the transparency of a blue-black ink.
"""
import math
import numpy as np, torch
import torch.nn.functional as Fn

DPI = 600
PX_PER_M = DPI / 0.0254
A4 = (0.210, 0.297)
R0, R1 = 1.6, 2.6            # nib footprint radius in px at zero and full pressure (0.07 mm .. 0.18 mm)
FLOW = 260.0                 # ink density per second under the nib; ~2 per pass at V_REF
V_REF = 0.05                 # m/s; slower than this deposits more per length
INK = torch.tensor([0.10, 0.14, 0.34])       # blue-black at full density
INK_THIN = torch.tensor([0.35, 0.42, 0.70])  # dilute ink
PAPER = torch.tensor([0.965, 0.955, 0.925])


class Page:
    def __init__(self, size=A4, device="cuda", seed=0):
        self.W, self.H = int(round(size[0] * PX_PER_M)), int(round(size[1] * PX_PER_M))
        self.dev = torch.device(device)
        self.D = torch.zeros(self.H, self.W, device=self.dev)
        g = torch.Generator(device=self.dev).manual_seed(seed)
        # paper: fibre texture with a slight horizontal grain, plus a low-frequency mottle
        n = torch.randn(self.H, self.W, device=self.dev, generator=g)
        fib = Fn.avg_pool2d(n[None, None], (1, 7), stride=1, padding=(0, 3))[0, 0]
        fib = Fn.avg_pool2d(fib[None, None], 3, stride=1, padding=1)[0, 0]
        low = Fn.interpolate(torch.randn(1, 1, self.H // 64 + 1, self.W // 64 + 1, device=self.dev, generator=g),
                             size=(self.H, self.W), mode="bicubic", align_corners=False)[0, 0]
        self.tex = (fib * 0.9 + low * 0.12)
        self.absorb = 1.0 + 0.25 * self.tex.clamp(-2, 2)     # where fibres are, ink takes more
        R = int(math.ceil(R0 + R1)) + 1
        oy, ox = torch.meshgrid(torch.arange(-R, R + 1, device=self.dev), torch.arange(-R, R + 1, device=self.dev), indexing="ij")
        self.off = torch.stack([oy.flatten(), ox.flatten()], 1)            # (K, 2)
        self.offd = (self.off.float() ** 2).sum(1).sqrt()

    def to_px(self, xy):
        """page metres (x right, y up from the bottom-left corner) -> (row, col) float."""
        col = xy[:, 0] * PX_PER_M
        row = self.H - xy[:, 1] * PX_PER_M
        return row, col

    def add(self, xy, p, dt, upsample=4):
        """Deposit a run of samples. xy (N,2) page metres, p (N,) pressure, dt seconds between samples."""
        if len(xy) < 2:
            return
        xy = torch.as_tensor(xy, dtype=torch.float32, device=self.dev)
        p = torch.as_tensor(p, dtype=torch.float32, device=self.dev)
        v = torch.cat([(xy[1:] - xy[:-1]).norm(dim=1) / dt, torch.zeros(1, device=self.dev)])
        v[-1] = v[-2]
        if upsample > 1:
            n = len(xy)
            t = torch.arange(n, device=self.dev, dtype=torch.float32)
            tu = torch.arange(0, n - 1, 1 / upsample, device=self.dev)
            i0 = tu.floor().long(); f = (tu - i0).unsqueeze(1)
            xy = xy[i0] * (1 - f) + xy[i0 + 1] * f
            p = p[i0] * (1 - f[:, 0]) + p[i0 + 1] * f[:, 0]
            v = v[i0] * (1 - f[:, 0]) + v[i0 + 1] * f[:, 0]
            dt = dt / upsample
        down = p > 0.10
        if down.sum() == 0:
            return
        xy, p, v = xy[down], p[down], v[down]
        row, col = self.to_px(xy)
        r = R0 + R1 * p
        amount = FLOW * dt * (V_REF / v.clamp(min=0.004)).clamp(max=4.0) * (0.35 + 0.65 * p)
        rr = row.unsqueeze(1) + self.off[:, 0]
        cc = col.unsqueeze(1) + self.off[:, 1]
        d = self.offd.unsqueeze(0)
        # soft-edged disc: full inside r-0.8, fading over one px
        w = ((r.unsqueeze(1) + 0.3 - d) / 1.1).clamp(0, 1)
        w = w * w * (3 - 2 * w)
        val = (w * amount.unsqueeze(1)).flatten()
        ri = rr.round().long().flatten(); ci = cc.round().long().flatten()
        ok = (ri >= 0) & (ri < self.H) & (ci >= 0) & (ci < self.W) & (val > 0)
        self.D.index_put_((ri[ok], ci[ok]), val[ok], accumulate=True)

    def render(self, r0=0, r1=None, c0=0, c1=None, out_size=None):
        """RGB float image (3, h, w) of the crop, optionally resized to out_size=(h, w)."""
        r1 = self.H if r1 is None else r1
        c1 = self.W if c1 is None else c1
        D = self.D[r0:r1, c0:c1][None, None]
        # bleed along the fibres: a little of the ink spreads sideways
        Db = Fn.avg_pool2d(Fn.pad(D, (2, 2, 1, 1), mode="replicate"), (3, 5), stride=1)
        dens = 1 - torch.exp(-(0.85 * D + 0.15 * Db)[0, 0] * self.absorb[r0:r1, c0:c1] * 1.6)
        dens = dens.clamp(0, 1)
        tex = self.tex[r0:r1, c0:c1]
        paper = PAPER.to(self.dev)[:, None, None] * (1 + 0.012 * tex)
        ink = INK_THIN.to(self.dev)[:, None, None] * (1 - dens) + INK.to(self.dev)[:, None, None] * dens
        img = paper * (1 - dens) + ink * dens
        if out_size is not None:
            if out_size[0] < img.shape[1]:
                img = Fn.interpolate(img[None], size=out_size, mode="area")[0]
            else:
                img = Fn.interpolate(img[None], size=out_size, mode="bicubic", align_corners=False)[0]
        return img.clamp(0, 1)

    def render_u8(self, *a, **k):
        return (self.render(*a, **k) * 255 + 0.5).byte().permute(1, 2, 0).cpu().numpy()


if __name__ == "__main__":
    import glyphs as G, time
    from PIL import Image
    pg = Page()
    t0 = time.time()
    y = 0.297 - 0.030
    for line in ["the quick brown fox jumps over the lazy dog,", "a hand that learned to write. 2026 (day 63)"]:
        t, xy, p = G.trajectory(G.layout(line, 0.0032, 0.022, y), 250, K=0.10)
        pg.add(xy, p, 1 / 250)
        y -= 0.0095
    torch.cuda.synchronize(); print("deposit", time.time() - t0)
    img = pg.render_u8(0, 900, 0, 4960)
    Image.fromarray(img).save("../scratch/ink_test.png")
    Image.fromarray(pg.render_u8(0, 3508 * 2, 0, 4960, out_size=(1754, 1240))).save("../scratch/ink_page.png")
    print("saved")
