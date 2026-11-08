"""
CPU timing for the native 16x16 sprite model. Standalone: needs sprite_model.py
(in the pixelgen package), torch, transformers, and the checkpoint.

    python tools/bench_fm_cpu.py --ckpt models/pixelgen_ema.pt --threads 4
    python tools/bench_fm_cpu.py --ckpt models/pixelgen_ema.pt --steps 10,20,30 --n 8 --save-grid bench.png

Reports load time, then per-sprite wall time (CLIP + denoising) for each step count.
Use the same machine and --threads as your s6 baseline for a fair comparison
(pipeline.py's default is cores // 2, i.e. physical cores).
"""
import argparse
import os
import platform
import sys
import time
from pathlib import Path

# Allow direct execution without editable install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

try:
    from pixelgen.sprite_model import SpriteModel
except ImportError:
    from sprite_model import SpriteModel

PROMPTS = [
    "small green slime", "red health potion", "steel sword with a blue gem", "knight in armor",
    "goblin warrior", "slice of pizza", "wooden shield", "blue dragon",
    "skeleton with a bow", "treasure chest", "red mushroom", "black cat",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/pixelgen_ema.pt")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--steps", default="10,20,30", help="comma list of Euler step counts")
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=6, help="sprites timed per step count")
    ap.add_argument("--save-grid", default="", help="save the last step-count's sprites as a PNG grid")
    args = ap.parse_args()

    import torch
    t0 = time.perf_counter()
    m = SpriteModel(args.ckpt, device="cpu", threads=args.threads)
    load_s = time.perf_counter() - t0
    params = sum(p.numel() for p in m.net.parameters()) / 1e6
    print(f"machine: {platform.processor() or platform.machine()}  torch {torch.__version__}  "
          f"threads {args.threads}\nUNet {params:.1f}M params, trained {m.step} steps\nload: {load_s:.1f}s")

    m.generate(PROMPTS[0], seed=0, steps=4, guidance=args.guidance)     # warm-up, excluded from stats

    imgs = []
    for steps in [int(s) for s in args.steps.split(",")]:
        tot, clip, unet, imgs = [], [], [], []
        for i in range(args.n):
            t = time.perf_counter()
            rgba, tm = m.generate(PROMPTS[i % len(PROMPTS)], seed=i, steps=steps, guidance=args.guidance)
            tot.append(time.perf_counter() - t); clip.append(tm["clip_s"]); unet.append(tm["unet_s"])
            imgs.append(rgba)
        tot = np.array(tot)
        print(f"steps {steps:>3}: p50 {np.percentile(tot, 50):.2f}s  p95 {np.percentile(tot, 95):.2f}s  "
              f"max {tot.max():.2f}s | mean clip {np.mean(clip):.3f}s  denoise {np.mean(unet):.2f}s "
              f"({np.mean(unet) / steps * 1000:.0f} ms/step)")

    if args.save_grid and imgs:
        from PIL import Image
        s, cols = 6, min(6, len(imgs))
        rows = (len(imgs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 16 * s, rows * 16 * s), (70, 70, 80))
        for i, a in enumerate(imgs):
            im = Image.fromarray(a, "RGBA").resize((16 * s, 16 * s), Image.Resampling.NEAREST)
            sheet.paste(im, ((i % cols) * 16 * s, (i // cols) * 16 * s), im)
        sheet.save(args.save_grid)
        print("saved", args.save_grid)


if __name__ == "__main__":
    main()
