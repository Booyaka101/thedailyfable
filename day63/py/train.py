"""Train the writing controller by backpropagation through the arm physics.

    python -u train.py --iters 600 --out ../out/policy

Targets are random fragments of cursive laid out with glyphs.py at random tempo and page position.
Checkpoints are saved on a sparse schedule so the practice sequence (what the hand wrote after 1, 2, 5, 10
... attempts) can be rendered afterwards.
"""
import argparse, os, random, time, json
import numpy as np, torch
import arm as A, control as C, glyphs as G

RATE = 250
WORDS = ("the of and to in is that it was for on are with as his they be at one have this from or had by "
         "hot word but what some we can out other were all there when up use your how said an each she which "
         "do their time if will way about many then them write would like so these her long make thing see him "
         "two has look more day could go come did number sound no most people my over know water than call first "
         "who may down side been now find any new work part take get place made live where after back little only "
         "round man year came show every good me give our under name very through just form sentence great think "
         "say help low line differ turn cause much mean before move right boy old too same tell does set three "
         "want air well also play small end put home read hand port large spell add even land here must big high "
         "such follow act why ask men change went light kind off need house picture try us again animal point "
         "mother world near build self earth father head stand own page should country found answer school grow "
         "study still learn plant cover food sun four between state keep eye never last let thought city tree "
         "cross farm hard start might story saw far sea draw left late run while press close night real life few "
         "north open seem together next white children begin got walk example ease paper group always music those "
         "both mark often letter until mile river car feet care second book carry took science eat room friend "
         "began idea fish mountain stop once base hear horse cut sure watch colour face wood main enough plain "
         "girl usual young ready above ever red list though feel talk bird soon body dog family direct pose leave "
         "song measure door product black short numeral class wind question happen complete ship area half rock "
         "order fire south problem piece told knew pass since top whole king street inch multiply nothing course "
         "stay wheel full force blue object decide surface deep moon island foot system busy test record boat "
         "common gold possible plane stead dry wonder laugh thousand ago ran check game shape equate hot miss "
         "brought heat snow tire bring yes distant fill east paint language among muscles owner fable sixty").split()
CAPS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIG = "0123456789"
PUN = ".,'?!-:;()"


def random_text(rng):
    n = rng.randint(1, 4)
    ws = [rng.choice(WORDS) for _ in range(n)]
    if rng.random() < 0.25:
        ws[0] = rng.choice(CAPS) + ws[0]
    if rng.random() < 0.15:
        ws.append("".join(rng.choice(DIG) for _ in range(rng.randint(1, 4))))
    if rng.random() < 0.3:
        ws[-1] += rng.choice(PUN)
    return " ".join(ws)


def make_pool(n, T, rng, ex=0.0032):
    """n target snippets of T samples: (n, T, 3) in metres and pressure."""
    pool = np.zeros((n, T, 3), np.float32)
    for i in range(n):
        while True:
            K = rng.uniform(0.06, 0.14)
            x0 = rng.uniform(0.05, 0.20)
            y0 = rng.uniform(0.40, 0.45)
            t, xy, p = G.trajectory(G.layout(random_text(rng), ex * rng.uniform(0.85, 1.15), x0, y0), RATE, K=K,
                                    dwell=rng.uniform(0.05, 0.3))
            if len(t) >= T:
                break
        s = rng.randint(0, len(t) - T) if rng.random() < 0.5 else 0
        pool[i, :, :2] = xy[s:s + T]
        pool[i, :, 2] = p[s:s + T]
    return torch.tensor(pool)


CKPT_ITERS = sorted(set([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300,
                         400, 500, 600, 800, 1000, 1200, 1500, 2000]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--T", type=float, default=2.4)
    ap.add_argument("--pool", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--effort", type=float, default=2.0)
    ap.add_argument("--noise", type=int, default=1)
    ap.add_argument("--gmax", type=float, default=200.0)
    ap.add_argument("--cap", type=float, default=15.0, help="per-episode rms cap in mm inside the loss")
    ap.add_argument("--out", default="../out/policy")
    ap.add_argument("--resume", default="")
    ap.add_argument("--seed", type=int, default=63)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    T = int(a.T * RATE)
    t0 = time.time()
    pool = make_pool(a.pool, T, rng).to(A.dev)
    print(f"pool {tuple(pool.shape)} in {time.time() - t0:.0f} s", flush=True)
    pol = C.Policy().to(A.dev)
    opt = torch.optim.Adam(pol.parameters(), lr=a.lr)
    start = 0
    if a.resume:
        ck = torch.load(a.resume, map_location=A.dev)
        pol.load_state_dict(ck["policy"]); opt.load_state_dict(ck["opt"]); start = ck["iter"] + 1
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters, eta_min=a.lr * 0.1)
    for _ in range(start):
        sched.step()
    arm = A.Arm(a.batch, noise=bool(a.noise))
    # a fixed test batch with its own noise seed: the learning curve without sampling noise
    ev_tgt = make_pool(32, T, random.Random(a.seed + 1)).to(A.dev)
    ev_arm = A.Arm(32, noise=True, gen=torch.Generator(device=A.dev))
    log = open(os.path.join(a.out, "log.txt"), "a")
    for it in range(start, a.iters + 1):
        if it in CKPT_ITERS:
            torch.save({"policy": pol.state_dict(), "opt": opt.state_dict(), "iter": it},
                       os.path.join(a.out, f"ckpt_{it:05d}.pt"))
            ev_arm.gen.manual_seed(a.seed)
            with torch.no_grad():
                ev = C.rollout(pol, ev_arm, ev_tgt, trunc=0)
            msg = f"eval it {it:4d} rms {ev['err2'].sqrt().item() * 1000:.3f} mm perr {ev['perr'].item():.4f} effort {ev['effort'].item():.4f}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if it == a.iters:
            break
        idx = torch.randint(0, pool.shape[0], (a.batch,), device=A.dev)
        tgt = pool[idx]
        t1 = time.time()
        out = C.rollout(pol, arm, tgt)
        # per-episode rms in mm rather than squared error: an episode that flails does not own the gradient
        track = ((out["err2_b"] * 1e6).clamp(max=a.cap ** 2) + 1.0).sqrt().mean() - 1.0
        loss = track + 0.5 * out["perr"] + a.effort * out["effort"]
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
        if torch.isfinite(gn) and gn < a.gmax:
            opt.step()
        else:
            print(f"skip step, gn {gn:.1f}", flush=True)
        sched.step()
        rms = out["err2"].sqrt().item() * 1000
        msg = (f"it {it:4d} loss {loss.item():.3f} rms {rms:.3f} mm perr {out['perr'].item():.3f} "
               f"effort {out['effort'].item():.4f} gn {gn:.2f} lr {sched.get_last_lr()[0]:.1e} {time.time() - t1:.1f}s")
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()
    torch.save({"policy": pol.state_dict(), "opt": opt.state_dict(), "iter": a.iters},
               os.path.join(a.out, "final.pt"))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
