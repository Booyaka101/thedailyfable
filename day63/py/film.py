"""The film: practice montage, then the letter written in real time, then the page.

    python film.py --letter ../out/letter --practice ../out/practice --out ../out/film.mp4
    python film.py --letter ../out/letter --practice ../out/practice --out ../out/short.mp4 --short

Main view: a 600 dpi crop of the page following the pen. Inset: the desk from above with the arm, the page
where it lies, and the eight muscle activations. Audio: pen.wav from sound.py under the letter scene.
"""
import argparse, json, os, subprocess, time
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import soundfile as sf
import ink

FPS = 60
FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_L = "C:/Windows/Fonts/segoeuil.ttf"
DESK = np.array([58, 52, 48], np.uint8)
INSET_BG = (72, 66, 61)
LINK = (225, 214, 196)
MUSCLE_NAMES = ["sh+", "sh-", "el+", "el-", "bi+", "bi-", "wr+", "wr-"]
A4_H_PX = int(round(0.297 * ink.PX_PER_M))
A4_W_PX = int(round(0.210 * ink.PX_PER_M))


def load_trace(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


class Encoder:
    def __init__(self, out, W, H):
        self.W, self.H = W, H
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
               "-i", "-", "-c:v", "hevc_nvenc", "-preset", "p5", "-rc", "vbr", "-b:v", "60M", "-maxrate", "90M",
               "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", out]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.n = 0

    def write(self, frame):
        assert frame.shape == (self.H, self.W, 3), frame.shape
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close(); self.p.wait()


def view(page, cx, cy, w, out_w, out_h):
    """Render the page region centred on (cx, cy) page px, w px wide, into an out_w x out_h frame on the desk."""
    h = w * out_h / out_w
    scale = out_w / w
    x0, y0 = cx - w / 2, cy - h / 2
    c0, c1 = int(max(0, np.floor(x0))), int(min(page.W, np.ceil(x0 + w)))
    r0, r1 = int(max(0, np.floor(y0))), int(min(page.H, np.ceil(y0 + h)))
    frame = np.empty((out_h, out_w, 3), np.uint8); frame[:] = DESK
    if c1 <= c0 or r1 <= r0:
        return frame
    ow, oh = int(round((c1 - c0) * scale)), int(round((r1 - r0) * scale))
    ox, oy = int(round((c0 - x0) * scale)), int(round((r0 - y0) * scale))
    ow, oh = min(ow, out_w - ox), min(oh, out_h - oy)
    if ow <= 0 or oh <= 0 or ox >= out_w or oy >= out_h:
        return frame
    img = page.render_u8(r0, r1, c0, c1, out_size=(oh, ow))
    frame[max(oy, 0):oy + oh, max(ox, 0):ox + ow] = img[max(-oy, 0):, max(-ox, 0):]
    return frame


def draw_pen(im, tip_px, direction, scale):
    """A pen held at the tip, leaning away along `direction` (unit vector in page px space)."""
    d = ImageDraw.Draw(im, "RGBA")
    L = 900 * scale
    x0, y0 = tip_px
    x1, y1 = x0 + direction[0] * L, y0 + direction[1] * L
    d.line([(x0 + direction[0] * 60 * scale, y0 + direction[1] * 60 * scale), (x1, y1)], fill=(28, 26, 30, 235), width=int(38 * scale) + 1)
    d.line([(x0, y0), (x0 + direction[0] * 70 * scale, y0 + direction[1] * 70 * scale)], fill=(170, 150, 110, 230), width=int(14 * scale) + 1)
    d.line([(x0 + direction[0] * 300 * scale, y0 + direction[1] * 300 * scale), (x1, y1)], fill=(255, 255, 255, 40), width=int(8 * scale) + 1)


class Inset:
    """Desk from above: shoulder at the bottom, the page where it lies, the arm, the muscle bars."""

    def __init__(self, w=440, h=620, bars_h=96):
        self.w, self.h, self.bars_h = w, h, bars_h
        self.s = h / 0.66                      # px per metre
        self.x_off = 0.06
        self.font = ImageFont.truetype(FONT, 20)
        self.font_s = ImageFont.truetype(FONT, 16)
        self.thumb = None

    def to_px(self, xy):
        return ((xy[0] + self.x_off) * self.s, self.h - 0.02 * self.s - xy[1] * self.s)

    def update_thumb(self, page):
        th = int(0.297 * self.s); tw = int(0.210 * self.s)
        self.thumb = Image.fromarray(page.render_u8(0, page.H, 0, page.W, out_size=(th, tw)))

    def draw(self, joints, a, offset, fat, t_s):
        im = Image.new("RGB", (self.w, self.h + self.bars_h), INSET_BG)
        d = ImageDraw.Draw(im, "RGBA")
        # page
        px, py = self.to_px((offset[0], offset[1] + 0.297))
        if self.thumb is not None:
            im.paste(self.thumb, (int(px), int(py)))
        d.rectangle([px, py, px + 0.210 * self.s, py + 0.297 * self.s], outline=(120, 110, 100, 255), width=1)
        # arm
        pts = [self.to_px(j) for j in joints]
        widths = [26, 20, 10]
        for i in range(3):
            d.line([pts[i], pts[i + 1]], fill=LINK, width=widths[i])
        for i, r in enumerate([14, 11, 7]):
            x, y = pts[i]
            d.ellipse([x - r, y - r, x + r, y + r], fill=(150, 140, 128), outline=(40, 36, 34))
        x, y = pts[3]
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(230, 60, 40))
        d.text((10, 8), "desk, from above", font=self.font_s, fill=(190, 182, 172))
        d.text((10, self.h - 30), f"{t_s:5.1f} s   fatigue {100 * fat:.1f}%", font=self.font_s, fill=(190, 182, 172))
        # muscle bars
        by = self.h + 10
        bw = (self.w - 20) / 8
        for i in range(8):
            x0 = 10 + i * bw
            d.rectangle([x0 + 3, by, x0 + bw - 3, by + 54], fill=(52, 48, 44))
            hgt = 54 * float(np.clip(a[i], 0, 1))
            col = (235, 90, 60) if i % 2 == 0 else (80, 150, 235)
            d.rectangle([x0 + 3, by + 54 - hgt, x0 + bw - 3, by + 54], fill=col)
            d.text((x0 + 8, by + 58), MUSCLE_NAMES[i], font=self.font_s, fill=(190, 182, 172))
        return np.asarray(im)


def card(W, H, lines, sizes, t_hold, enc, fade=0.6, sub_color=(170, 162, 150)):
    im = Image.new("RGB", (W, H), tuple(int(v) for v in DESK))
    d = ImageDraw.Draw(im)
    y = H // 2 - sum(sizes) * 0.7
    for text, sz in zip(lines, sizes):
        f = ImageFont.truetype(FONT_L if sz > 60 else FONT, sz)
        tw = d.textlength(text, font=f)
        d.text(((W - tw) / 2, y), text, font=f, fill=(232, 224, 210) if sz > 60 else sub_color)
        y += sz * 1.5
    base = np.asarray(im).astype(np.float32)
    n = int(t_hold * FPS); nf = int(fade * FPS)
    desk = DESK.astype(np.float32)
    for i in range(n):
        k = min(1.0, i / nf, (n - 1 - i) / nf)
        enc.write((desk + (base - desk) * k).astype(np.uint8))


def frame_samples(f, rate):
    return int(round(f * rate / FPS)), int(round((f + 1) * rate / FPS))


def deposit(page, tr, i0, i1):
    j0 = max(i0 - 1, 0)
    if i1 - j0 < 2:
        return
    xy = tr["tip"][j0:i1] - tr["offset"][j0:i1]
    page.add(xy, tr["p"][j0:i1], 1.0 / float(tr["rate"]))


def pen_dir(tr, i):
    """Unit vector from the tip toward the wrist, in page px (y down), leaning a little to the right."""
    j = tr["joints"][i]
    v = j[2] - j[3]
    v = np.array([v[0], -v[1]]); v = v / (np.linalg.norm(v) + 1e-9)
    v = v * 0.75 + np.array([0.55, -0.35])
    return v / np.linalg.norm(v)


class Camera:
    def __init__(self, tau=0.45):
        self.c = None; self.tau = tau

    def follow(self, target, dt):
        if self.c is None:
            self.c = np.array(target, float)
        k = 1 - np.exp(-dt / self.tau)
        self.c = self.c + (np.array(target, float) - self.c) * k
        return self.c


def letter_scene(enc, page, tr, W, H, inset, zoom_w, pen_anchor, t_start=0.0, t_end=None, fade_in=1.0, speed=1):
    """Render the letter trace from t_start to t_end (s) in real time; returns the last camera centre."""
    rate = int(tr["rate"]); N = len(tr["tip"])
    t_end = N / rate if t_end is None else t_end
    cam = Camera()
    f0, f1 = int(t_start * FPS), int(t_end * FPS)
    # deposit anything before t_start without drawing
    i_pre = int(t_start * rate)
    if i_pre > 1:
        page.add(tr["tip"][:i_pre] - tr["offset"][:i_pre], tr["p"][:i_pre], 1.0 / rate)
    scale = W / zoom_w
    frames = list(range(f0, f1, speed))
    for n, f in enumerate(frames):
        i0, i1 = frame_samples(f, rate); i1 = min(i1 + (speed - 1) * int(rate / FPS), N)
        i0 = min(i0, N - 1); i1 = max(i1, i0 + 1)
        deposit(page, tr, i0, i1)
        i = i1 - 1
        pxy = tr["tip"][i] - tr["offset"][i]
        col, row = pxy[0] * ink.PX_PER_M, page.H - pxy[1] * ink.PX_PER_M
        h = zoom_w * H / W
        tgt = (col + (0.5 - pen_anchor[0]) * zoom_w, row + (0.5 - pen_anchor[1]) * h)
        tgt = (float(np.clip(tgt[0], zoom_w / 2, page.W - zoom_w / 2)), float(np.clip(tgt[1], h / 2, page.H - h / 2)))
        c = cam.follow(tgt, 1.0 / FPS)
        frame = view(page, c[0], c[1], zoom_w, W, H)
        im = Image.fromarray(frame)
        draw_pen(im, ((col - (c[0] - zoom_w / 2)) * scale, (row - (c[1] - h / 2)) * scale), pen_dir(tr, i), scale)
        frame = np.asarray(im).copy()
        if inset is not None:
            if n % 30 == 0:
                inset.update_thumb(page)
            ins = inset.draw(tr["joints"][i], tr["a"][i], tr["offset"][i], tr["fat"][i].max(), i / rate)
            ih, iw = ins.shape[:2]
            frame[H - ih - 40:H - 40, W - iw - 40:W - 40] = ins
        if fade_in and n < fade_in * FPS:
            k = n / (fade_in * FPS)
            frame = (DESK + (frame.astype(np.float32) - DESK) * k).astype(np.uint8)
        enc.write(frame)
    return cam.c


def zoom_out(enc, page, c_from, w_from, W, H, secs=4.0, hold=6.0, inset=None):
    """Pull back from the writing crop to the whole page, then hold."""
    cx1, cy1 = page.W / 2, page.H / 2
    w1 = page.H * 1.08 * W / H
    n = int(secs * FPS)
    for i in range(n):
        s = i / (n - 1); s = 0.5 - 0.5 * np.cos(np.pi * s)
        lw = np.exp(np.log(w_from) * (1 - s) + np.log(w1) * s)
        cx = c_from[0] + (cx1 - c_from[0]) * s; cy = c_from[1] + (cy1 - c_from[1]) * s
        enc.write(view(page, cx, cy, lw, W, H))
    last = view(page, cx1, cy1, w1, W, H)
    for i in range(int(hold * FPS)):
        enc.write(last)
    return last


def practice_scene(enc, page, practice_dir, W, H, speed=3, hold=0.6, subset=None, per_max=None):
    """Each checkpoint writes the phrase, sped up, with a caption saying which attempt it is."""
    summ = json.load(open(os.path.join(practice_dir, "summary.json")))
    attempts = summ["attempts"] if subset is None else [a for a in summ["attempts"] if a["iter"] in subset]
    font = ImageFont.truetype(FONT, 44); font_s = ImageFont.truetype(FONT, 30)
    for a in attempts:
        tr = load_trace(os.path.join(practice_dir, a["file"]))
        rate = int(tr["rate"]); N = len(tr["tip"])
        page.D.zero_()
        # fixed camera on the phrase
        pxy = tr["target"][:, :2] - tr["offset"]
        down = tr["target"][:, 2] > 0.1
        cx = (pxy[down, 0].min() + pxy[down, 0].max()) / 2 * ink.PX_PER_M
        cy = page.H - (pxy[down, 1].min() + pxy[down, 1].max()) / 2 * ink.PX_PER_M
        zoom_w = max((pxy[down, 0].max() - pxy[down, 0].min()) * ink.PX_PER_M * 1.35, 1800)
        nfr = int(np.ceil(N / rate * FPS / speed))
        if per_max is not None:
            nfr = min(nfr, int(per_max * FPS))
        scale = W / zoom_w
        h = zoom_w * H / W
        cap = f"attempt {a['iter'] + 1}" if a["iter"] < 10 ** 6 else ""
        for f in range(nfr + int(hold * FPS)):
            ff = min(f, nfr - 1)
            i0 = min(int(ff * speed * rate / FPS), N - 1); i1 = min(int((ff + 1) * speed * rate / FPS), N)
            if i1 > i0:
                deposit(page, tr, i0, i1)
            i = max(i1 - 1, 0)
            frame = view(page, cx, cy, zoom_w, W, H)
            im = Image.fromarray(frame)
            if f < nfr:
                p = tr["tip"][i] - tr["offset"][i]
                col, row = p[0] * ink.PX_PER_M, page.H - p[1] * ink.PX_PER_M
                draw_pen(im, ((col - (cx - zoom_w / 2)) * scale, (row - (cy - h / 2)) * scale), pen_dir(tr, i), scale)
            d = ImageDraw.Draw(im)
            d.text((60, 44), cap, font=font, fill=(60, 56, 58))
            d.text((60, 104), f"error {a['rms_mm']:.2f} mm", font=font_s, fill=(120, 112, 108))
            enc.write(np.asarray(im))


def build_audio(out_wav, pen_wav, lead_s, tail_s, total_s):
    y, fs = sf.read(pen_wav, dtype="float32")
    n_total = int(total_s * fs)
    audio = np.zeros((n_total, 2), np.float32)
    i0 = int(lead_s * fs)
    m = min(len(y), n_total - i0)
    audio[i0:i0 + m] = y[:m]
    rng = np.random.default_rng(1)
    audio += (rng.standard_normal((n_total, 2)) * 0.0015).astype(np.float32)   # room floor
    nf = int(1.5 * fs)
    audio[-nf:] *= np.linspace(1, 0, nf)[:, None]
    sf.write(out_wav, audio, fs, subtype="PCM_24")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letter", required=True)
    ap.add_argument("--practice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--short", action="store_true")
    ap.add_argument("--t_end", type=float, default=None, help="stop the letter early (for tests)")
    ap.add_argument("--attempts", default="0,1,2,3,4,5,6,8,10,12,15,20,25,30", help="checkpoint iterations shown as practice")
    a = ap.parse_args()
    attempts = [int(x) for x in a.attempts.split(",")]
    t0 = time.time()
    tr = load_trace(os.path.join(a.letter, "trace.npz"))
    W, H = (1080, 1920) if a.short else (2560, 1440)
    page = ink.Page()
    tmp = a.out + ".video.mp4"
    enc = Encoder(tmp, W, H)
    if not a.short:
        card(W, H, ["A hand that learned to write", "a simulated arm, eight muscles, one letter"], [96, 36], 4.0, enc)
        practice_scene(enc, page, a.practice, W, H, subset=attempts)
        card(W, H, ["the letter", "written in one take, in real time"], [72, 36], 3.0, enc)
        lead_s = enc.n / FPS
        page.D.zero_()
        inset = Inset()
        c = letter_scene(enc, page, tr, W, H, inset, zoom_w=2560, pen_anchor=(0.42, 0.56), t_end=a.t_end)
        for _ in range(int(2.0 * FPS)):
            enc.write(view(page, c[0], c[1], 2560, W, H))
        zoom_out(enc, page, c, 2560, W, H)
        card(W, H, ["day 63", "thedailyfable"], [72, 36], 4.0, enc)
    else:
        card(W, H, ["a hand that", "learned to write"], [96, 96], 2.5, enc)
        practice_scene(enc, page, a.practice, W, H, speed=4, hold=0.3, subset=[i for i in (0, 1, 2, 5, 10, 20, 30) if i in attempts], per_max=2.0)
        lead_s = enc.n / FPS
        page.D.zero_()
        c = letter_scene(enc, page, tr, W, H, None, zoom_w=1500, pen_anchor=(0.45, 0.5), t_end=22.0)
        zoom_out(enc, page, c, 1500, W, H, secs=3.0, hold=1.0)
        # skip ahead: the rest of the page appears, written
        page.add(tr["tip"] - tr["offset"], tr["p"], 1.0 / float(tr["rate"]))
        zoom_out(enc, page, (page.W / 2, page.H / 2), page.H * 1.08 * W / H, W, H, secs=0.5, hold=6.0)
    enc.close()
    total_s = enc.n / FPS
    wav = a.out + ".audio.wav"
    build_audio(wav, os.path.join(a.letter, "pen.wav"), lead_s, 0, total_s)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "256k",
                    "-shortest", a.out], check=True)
    os.remove(tmp); os.remove(wav)
    print(f"wrote {a.out}: {total_s:.1f} s, {enc.n} frames in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
