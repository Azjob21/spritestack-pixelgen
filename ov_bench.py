"""
OpenVINO export + benchmark for SpriteStack PixelGen.  (UNTESTED on the target machine.)

Put this file in the repo root (next to benchmarks\\) and run from there.

    # 1) one-time export: fuses the LoRA, converts to OpenVINO with static 512x512 shapes
    python ov_bench.py export --lora-scale 0.9 --out models\\ov-pixelart

    # 2) benchmark: same prompts / seed / post-processing / output files as `pixelgen benchmark`
    python ov_bench.py bench --model models\\ov-pixelart --steps 6 --guidance 2.5 --tiny-vae ^
        --prompts benchmarks\\hard3.txt --save-raw --out eval_ov/

Design notes
  * The UNet (86% of the time) and the text encoder run in OpenVINO. The VAE decode stays in
    PyTorch (TAESD, ~1s), so no VAE export is needed.
  * The model must be exported at the batch size it will be used at. With guidance > 1 the
    pipeline runs the UNet at batch 2 (cond + uncond); optimum handles that for batch_size=1.
  * Re-export if you change lora_scale (it is fused into the weights).
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

from pixelgen.pipeline import Config, _postprocess, build_prompts, get_config, load_pipeline

SIZE = 512


def cmd_export(a: argparse.Namespace) -> None:
    from optimum.intel import OVStableDiffusionPipeline

    out = Path(a.out)
    tmp = out.with_name(out.name + "_fused_torch")
    # Reuse the repo's loader: base model + LoRA fused at the requested scale (default VAE).
    cfg = get_config("s4", lora_scale=a.lora_scale)
    cfg = Config(**{**cfg.__dict__, "name": "export", "tiny_vae": False})
    pipe = load_pipeline(cfg)
    # The LoRA is already fused into the weights. Remove the leftover PEFT wrapper layers,
    # otherwise save_pretrained writes `base_layer`/`lora_A` keys that can't be reloaded.
    pipe.unload_lora_weights()
    bad = [k for k in pipe.unet.state_dict() if "lora" in k or "base_layer" in k]
    if bad:
        sys.exit(f"UNet still has LoRA wrapper keys (e.g. {bad[0]}); refusing to save a broken export.")
    print(f"Saving fused PyTorch pipeline to {tmp} ...")
    pipe.save_pretrained(str(tmp))

    print("Exporting to OpenVINO (a few minutes) ...")
    t0 = time.perf_counter()
    ov = OVStableDiffusionPipeline.from_pretrained(str(tmp), export=True, compile=False)
    ov.reshape(batch_size=1, height=a.size, width=a.size, num_images_per_prompt=1)
    ov.save_pretrained(str(out))
    print(f"Done in {time.perf_counter() - t0:.0f}s -> {out}")
    print(f"(You can delete {tmp} afterwards.)")


def _load_ov(model: Path, tiny_vae: bool, threads: int, size: int):
    import torch
    from optimum.intel import OVStableDiffusionPipeline
    from diffusers import AutoencoderTiny

    ov = OVStableDiffusionPipeline.from_pretrained(
        str(model), compile=False,
        ov_config={"PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": threads},
    )
    ov.reshape(batch_size=1, height=size, width=size, num_images_per_prompt=1)
    ov.compile()
    if tiny_vae:
        src = Path("models/taesd")
        vae = AutoencoderTiny.from_pretrained(str(src) if src.is_dir() else "madebyollin/taesd",
                                              torch_dtype=torch.float32).eval()
    else:
        vae = None   # use the pipeline's own OpenVINO VAE decoder
    return ov, vae


def _gen(ov, vae, cfg: Config, prompt: str, w: int, h: int, seed: int):
    import torch

    pos, neg = build_prompts(cfg, prompt)
    rng = np.random.RandomState(seed)   # optimum-intel pipelines take a numpy RandomState
    t0 = time.perf_counter()
    if vae is not None:
        lat = ov(pos, negative_prompt=neg, width=cfg.gen_size, height=cfg.gen_size,
                 num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
                 generator=rng, output_type="latent").images
        t1 = time.perf_counter()
        lat = torch.from_numpy(np.asarray(lat)).float()
        with torch.inference_mode():
            dec = vae.decode(lat / vae.config.scaling_factor, return_dict=False)[0]
        pil = ov.image_processor.postprocess(dec, output_type="pil")[0]
    else:
        pil = ov(pos, negative_prompt=neg, width=cfg.gen_size, height=cfg.gen_size,
                 num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
                 generator=rng).images[0]
        t1 = time.perf_counter()
    t2 = time.perf_counter()
    img = _postprocess(cfg, pil, w, h)
    t3 = time.perf_counter()
    return img, pil, {"unet_s": t1 - t0, "decode_s": t2 - t1, "post_s": t3 - t2, "total_s": t3 - t0}


def _sheet(images, cols=6, cell=128):
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), (70, 70, 80))
    for i, im in enumerate(images):
        tile = Image.new("RGB", (cell, cell), (70, 70, 80))
        s = max(1, cell // max(im.size))
        big = im.convert("RGBA").resize((im.width * s, im.height * s), Image.Resampling.NEAREST)
        tile.paste(big, ((cell - big.width) // 2, (cell - big.height) // 2), big)
        sheet.paste(tile, ((i % cols) * cell, (i // cols) * cell))
    return sheet


def cmd_bench(a: argparse.Namespace) -> None:
    prompts = [l.strip() for l in Path(a.prompts).read_text().splitlines()
               if l.strip() and not l.strip().startswith("#")]
    if a.limit:
        prompts = prompts[: a.limit]
    w, h = (int(x) for x in a.size.lower().split("x"))
    cfg = get_config("s4", steps=a.steps, guidance=a.guidance, tiny_vae=True if a.tiny_vae else None,
                     bg_tol=a.bg_tol, edge_thr=a.edge_thr, prompt_style=a.prompt_style)
    out_dir = Path(a.out) / ("ov_" + cfg.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ov, vae = _load_ov(Path(a.model), cfg.tiny_vae, a.threads, cfg.gen_size)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    _gen(ov, vae, cfg, prompts[0], w, h, a.seed)   # warm-up, excluded from stats
    warm_s = time.perf_counter() - t0
    print(f"[ov_{cfg.name}] load {load_s:.1f}s, warm-up gen {warm_s:.1f}s")

    rows, images = [], []
    for i, prompt in enumerate(prompts, 1):
        img, raw, t = _gen(ov, vae, cfg, prompt, w, h, a.seed)
        images.append(img)
        stem = f"{i:03d}_{prompt.replace(' ', '_')[:40]}"
        img.save(out_dir / f"{stem}.png")
        if a.save_raw:
            (out_dir / "raw").mkdir(exist_ok=True)
            raw.save(out_dir / "raw" / f"{stem}.png")
        rows.append({"idx": i, "prompt": prompt, "seed": a.seed, **{k: f"{v:.4f}" for k, v in t.items()}})
        print(f"  [{i}/{len(prompts)}] {t['total_s']:.1f}s  {prompt!r}")

    with open(out_dir / "timings.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    _sheet(images).save(out_dir / "sheet.png")
    tot = np.array([float(r["total_s"]) for r in rows])
    st = {k: np.mean([float(r[k]) for r in rows]) for k in ("unet_s", "decode_s", "post_s")}
    summary = (
        f"variant: ov_{cfg.name}\nconfig: {cfg}\nsize: {w}x{h}  seed: {a.seed}  n: {len(rows)}\n"
        f"backend: openvino  threads: {a.threads}\n"
        f"machine: {platform.processor() or platform.machine()}  python {platform.python_version()}\n"
        f"load_s: {load_s:.2f}  warmup_s: {warm_s:.2f}\n"
        f"p50: {np.percentile(tot, 50):.2f}s  p95: {np.percentile(tot, 95):.2f}s  "
        f"min: {tot.min():.2f}s  max: {tot.max():.2f}s\n"
        f"mean unet: {st['unet_s']:.2f}s  decode: {st['decode_s']:.2f}s  post: {st['post_s']:.3f}s\n"
    )
    (out_dir / "summary.txt").write_text(summary)
    print("\n" + summary)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--out", default="models/ov-pixelart")
    e.add_argument("--lora-scale", type=float, default=0.9)
    e.add_argument("--size", type=int, default=SIZE)
    b = sub.add_parser("bench")
    b.add_argument("--model", default="models/ov-pixelart")
    b.add_argument("--prompts", default="benchmarks/hard3.txt")
    b.add_argument("--size", default="64x64")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--steps", type=int, default=6)
    b.add_argument("--guidance", type=float, default=2.5)
    b.add_argument("--tiny-vae", action="store_true")
    b.add_argument("--bg-tol", type=float, default=None)
    b.add_argument("--edge-thr", type=float, default=None)
    b.add_argument("--prompt-style", default=None)
    b.add_argument("--threads", type=int, default=4, help="physical cores on an i7-1065G7 = 4")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--save-raw", action="store_true")
    b.add_argument("--out", default="eval_ov")
    a = p.parse_args()
    {"export": cmd_export, "bench": cmd_bench}[a.cmd](a)


if __name__ == "__main__":
    main()
