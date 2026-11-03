"""
CLI entry point for spritestack-pixelgen.

Usage:
    python -m pixelgen.cli generate "small slime enemy" --size 32x32 --out slime.png
    python -m pixelgen.cli benchmark --prompts benchmarks/prompts.txt --out eval/v0/
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a single sprite and save it."""
    from .pipeline import generate

    w, h = (int(x) for x in args.size.split("x"))
    seed = args.seed if args.seed is not None else None

    print(f"Generating {w}×{h} sprite for: {args.prompt!r}")
    t0 = time.perf_counter()
    img = generate(prompt=args.prompt, width=w, height=h, seed=seed)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.2f}s")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out))
    print(f"Saved to {out}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run the benchmark suite over a prompt file."""
    from .pipeline import generate, load_pipeline

    prompts_file = Path(args.prompts)
    if not prompts_file.exists():
        print(f"Prompts file not found: {prompts_file}", file=sys.stderr)
        sys.exit(1)

    prompts = [
        line.strip()
        for line in prompts_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    w, h = (int(x) for x in args.size.split("x"))
    seed = args.seed

    # Warm up the pipeline
    print("Loading pipeline (warm-up)...")
    t0 = time.perf_counter()
    load_pipeline()
    warmup_time = time.perf_counter() - t0
    print(f"Pipeline loaded in {warmup_time:.2f}s")

    timings: list[float] = []
    for i, prompt in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {prompt!r}", end=" ", flush=True)
        t0 = time.perf_counter()
        img = generate(prompt=prompt, width=w, height=h, seed=seed)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        print(f"→ {elapsed:.2f}s")

        safe_name = prompt.replace(" ", "_")[:40]
        img.save(str(out_dir / f"{i:03d}_{safe_name}.png"))

    # Report
    timings.sort()
    n = len(timings)
    p50 = timings[n // 2] if n else 0
    p95 = timings[int(n * 0.95)] if n else 0
    print(f"\n--- Benchmark results ({n} prompts, {w}×{h}, seed={seed}) ---")
    print(f"  Warm-up:  {warmup_time:.2f}s")
    print(f"  p50:      {p50:.2f}s")
    print(f"  p95:      {p95:.2f}s")
    if timings:
        print(f"  min:      {timings[0]:.2f}s")
        print(f"  max:      {timings[-1]:.2f}s")
    print(f"  Samples saved to {out_dir}/")

    # Write results to a file
    results_file = out_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write(f"Benchmark: {n} prompts, {w}x{h}, seed={seed}\n")
        f.write(f"Hardware: Intel Core i7-1065G7, CPU-only\n")
        f.write(f"Warm-up: {warmup_time:.2f}s\n")
        f.write(f"p50: {p50:.2f}s\n")
        f.write(f"p95: {p95:.2f}s\n")
        if timings:
            f.write(f"min: {timings[0]:.2f}s\n")
            f.write(f"max: {timings[-1]:.2f}s\n")
        f.write(f"\nPer-prompt timings:\n")
        for prompt_text, t in zip(prompts, timings):
            f.write(f"  {t:.2f}s  {prompt_text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pixelgen",
        description="SpriteStack PixelGen — text-to-pixel-art pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = sub.add_parser("generate", help="Generate a single sprite")
    gen.add_argument("prompt", help="Text description of the sprite")
    gen.add_argument("--size", default="32x32", help="WxH (default: 32x32)")
    gen.add_argument("--seed", type=int, default=None, help="RNG seed (default: random)")
    gen.add_argument("--out", default="output.png", help="Output file path")

    # benchmark
    bench = sub.add_parser("benchmark", help="Run benchmark over a prompt set")
    bench.add_argument("--prompts", default="benchmarks/prompts.txt", help="Prompts file")
    bench.add_argument("--size", default="32x32", help="WxH (default: 32x32)")
    bench.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducibility")
    bench.add_argument("--out", default="eval/v0", help="Output directory")

    args = parser.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
