"""Write the same phrase with every saved checkpoint, so the practice can be watched.

    python practice.py --policies ../out/policy --text "the quick brown fox" --out ../out/practice

Each checkpoint writes the phrase once on a fresh page region with the same noise seed. Saves one
trace_NNNNN.npz per checkpoint (same layout as write.py) and a summary json with pen-down rms per attempt.
"""
import argparse, glob, json, os, re
import numpy as np, torch
import arm as A, control as C, glyphs as G, write as W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", required=True)
    ap.add_argument("--text", default="the quick brown fox")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    target, offs, lidx = W.build_target([a.text], a.K, np.random.RandomState(a.seed))
    files = sorted(glob.glob(os.path.join(a.policies, "ckpt_*.pt")))
    summary = []
    pol = C.Policy().to(A.dev)
    for f in files:
        it = int(re.search(r"ckpt_(\d+)", f).group(1))
        pol.load_state_dict(torch.load(f, map_location=A.dev)["policy"])
        tr = W.run(pol, target, seed=a.seed, fatigue=False)
        n = len(tr["tip"])
        err = np.linalg.norm(tr["tip"] - target[:n, :2], axis=1) * 1000
        down = target[:n, 2] > 0.1
        rms = float(np.sqrt((err[down] ** 2).mean()))
        summary.append({"iter": it, "rms_mm": rms, "file": f"trace_{it:05d}.npz"})
        print(f"iter {it:5d} pen-down rms {rms:.3f} mm", flush=True)
        np.savez_compressed(os.path.join(a.out, f"trace_{it:05d}.npz"), target=target[:n], offset=offs[:n], line=lidx[:n],
                            rate=W.RATE, **tr)
    json.dump({"text": a.text, "K": a.K, "seed": a.seed, "attempts": summary}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
