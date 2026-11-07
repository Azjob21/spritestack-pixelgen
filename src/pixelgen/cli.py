"""
CLI for spritestack-pixelgen.

    python -m pixelgen generate "small slime enemy" --size 32x32 --out slime.png
    python -m pixelgen benchmark --variant v0,e1,e2,e3 --out eval/
    python -m pixelgen benchmark --variant e2 --steps 6      # single-variable ablation
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .pipeline import DEFAULT_VARIANT, VARIANTS, generate, get_config, load_pipeline


def _size(s: str) -> tuple[int, int]:
    w, h = (int(x) for x in s.lower().split("x"))
    return w, h


def _overrides(args: argparse.Namespace) -> dict:
    return dict(steps=args.steps, guidance=args.guidance, gen_size=args.gen_size,
                palette=args.palette, outline=True if args.outline else None,
                tiny_vae=True if args.tiny_vae else None, bg_tol=args.bg_tol, edge_thr=args.edge_thr,
                lora_scale=args.lora_scale, prompt_style=args.prompt_style)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--gen-size", type=int, default=None)
    p.add_argument("--palette", default=None, help="db32 | pico8 | endesga32")
    p.add_argument("--outline", action="store_true")
    p.add_argument("--tiny-vae", action="store_true", help="use TAESD for decoding")
    p.add_argument("--bg-tol", type=float, default=None, help="backdrop keying tolerance")
    p.add_argument("--edge-thr", type=float, default=None,
                   help="edge threshold for gradient-aware keying (0 = off, default 2.0)")
    p.add_argument("--lora-scale", type=float, default=None, help="style LoRA strength")
    p.add_argument("--prompt-style", default=None, choices=["v0", "v0t", "clean"])
    p.add_argument("--save-raw", action="store_true", help="also save the raw model output")


def cmd_generate(args: argparse.Namespace) -> None:
    w, h = _size(args.size)
    print(f"Generating {w}x{h} [{args.variant}] for: {args.prompt!r}")
    res = generate(args.prompt, w, h, seed=args.seed, config=args.variant, **_overrides(args))
    t = res.timings
    print(f"seed={res.seed}  unet={t['unet_s']:.1f}s  decode={t['decode_s']:.2f}s  "
          f"post={t['post_s']:.3f}s  total={t['total_s']:.1f}s")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.image.save(out)
    if args.save_raw and res.raw is not None:
        res.raw.save(out.with_name(out.stem + "_raw.png"))
    print(f"Saved to {out}")


def _contact_sheet(images: list[Image.Image], cols: int = 6, cell: int = 128) -> Image.Image:
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (70, 70, 80))
    for i, im in enumerate(images):
        tile = Image.new("RGB", (cell, cell), (70, 70, 80))
        # checkerboard so transparency is visible
        px = tile.load()
        for y in range(cell):
            for x in range(cell):
                if ((x // 8) + (y // 8)) % 2:
                    px[x, y] = (90, 90, 100)
        s = max(1, cell // max(im.size))
        big = im.convert("RGBA").resize((im.width * s, im.height * s), Image.Resampling.NEAREST)
        tile.paste(big, ((cell - big.width) // 2, (cell - big.height) // 2), big)
        sheet.paste(tile, ((i % cols) * cell, (i // cols) * cell))
    return sheet


def cmd_benchmark(args: argparse.Namespace) -> None:
    prompts_file = Path(args.prompts)
    if not prompts_file.exists():
        sys.exit(f"Prompts file not found: {prompts_file}")
    prompts = [l.strip() for l in prompts_file.read_text().splitlines()
               if l.strip() and not l.strip().startswith("#")]
    if args.limit:
        prompts = prompts[: args.limit]
    w, h = _size(args.size)
    variants = [v.strip() for v in args.variant.split(",") if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            sys.exit(f"Unknown variant {v!r}. Available: {sorted(VARIANTS)}")

    root = Path(args.out)
    for v in variants:
        cfg = get_config(v, **_overrides(args))
        out_dir = root / cfg.name
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        load_pipeline(cfg)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        generate(prompts[0], w, h, seed=args.seed, config=cfg)   # warm-up, excluded from stats
        warm_s = time.perf_counter() - t0
        print(f"\n[{cfg.name}] load {load_s:.1f}s, warm-up gen {warm_s:.1f}s")

        rows, images = [], []
        for i, prompt in enumerate(prompts, 1):
            res = generate(prompt, w, h, seed=args.seed, config=cfg)
            images.append(res.image)
            stem = f"{i:03d}_{prompt.replace(' ', '_')[:40]}"
            res.image.save(out_dir / f"{stem}.png")
            if args.save_raw and res.raw is not None:
                (out_dir / "raw").mkdir(exist_ok=True)
                res.raw.save(out_dir / "raw" / f"{stem}.png")
            rows.append({"idx": i, "prompt": prompt, "seed": res.seed, **{k: f"{v:.4f}" for k, v in res.timings.items()}})
            print(f"  [{i}/{len(prompts)}] {res.timings['total_s']:.1f}s  {prompt!r}")

        with open(out_dir / "timings.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        _contact_sheet(images).save(out_dir / "sheet.png")

        tot = np.array([float(r["total_s"]) for r in rows])
        stage = {k: np.mean([float(r[k]) for r in rows]) for k in ("unet_s", "decode_s", "post_s")}
        opaque = [float(np.asarray(im.split()[3]).min() == 0) for im in images]
        summary = (
            f"variant: {cfg.name}\nconfig: {cfg}\nsize: {w}x{h}  seed: {args.seed}  n: {len(rows)}\n"
            f"machine: {platform.processor() or platform.machine()}  python {platform.python_version()}\n"
            f"load_s: {load_s:.2f}  warmup_s: {warm_s:.2f}\n"
            f"p50: {np.percentile(tot, 50):.2f}s  p95: {np.percentile(tot, 95):.2f}s  "
            f"min: {tot.min():.2f}s  max: {tot.max():.2f}s\n"
            f"mean unet: {stage['unet_s']:.2f}s  decode: {stage['decode_s']:.2f}s  post: {stage['post_s']:.3f}s\n"
            f"transparency rate (any transparent pixel): {np.mean(opaque):.0%}\n"
        )
        (out_dir / "summary.txt").write_text(summary)
        print("\n" + summary)


def main() -> None:
    parser = argparse.ArgumentParser(prog="pixelgen", description="SpriteStack PixelGen")
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Generate a single sprite")
    g.add_argument("prompt")
    g.add_argument("--size", default="32x32")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--variant", default=DEFAULT_VARIANT, choices=sorted(VARIANTS))
    g.add_argument("--out", default="output.png")
    _add_common(g)

    b = sub.add_parser("benchmark", help="Run prompts through one or more variants")
    b.add_argument("--prompts", default="benchmarks/prompts.txt")
    b.add_argument("--size", default="64x64")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--variant", default="v0", help="comma list, e.g. v0,e1,e2")
    b.add_argument("--limit", type=int, default=0, help="only first N prompts")
    b.add_argument("--out", default="eval")
    _add_common(b)

    args = parser.parse_args()
    {"generate": cmd_generate, "benchmark": cmd_benchmark}[args.command](args)


if __name__ == "__main__":
    main()
