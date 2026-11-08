"""
Prompt x seed contact sheet: one row per prompt, one column per seed.

    python tools/prompt_grid.py --ckpt models/pixelgen_ema.pt --steps 10 --guidance 2.0 --seeds 6 \
        --prompts "blue dragon" "red dragon" "fire dragon" "small red dragon" --out grid.png

Add --guidance-sweep 1.5,2.0,3.0,4.5 to render the same prompts at several guidance values
(one sheet per value, saved as grid_g<value>.png).
"""
import argparse
import sys
from pathlib import Path

# Allow direct execution without editable install
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image, ImageDraw

try:
    from pixelgen.sprite_model import SpriteModel
except ImportError:
    from sprite_model import SpriteModel


def make_sheet(rows: list[list[np.ndarray]], labels: list[str], scale: int = 6, label_w: int = 190) -> Image.Image:
    """rows[i][j] = uint8 RGBA (16,16,4) for prompt i, seed j."""
    n_r, n_c = len(rows), len(rows[0])
    S = rows[0][0].shape[0] * scale
    sheet = Image.new("RGB", (label_w + n_c * S, n_r * S), (70, 70, 80))
    d = ImageDraw.Draw(sheet)
    yy, xx = np.mgrid[0:S, 0:S]
    chk = np.where(((yy // scale + xx // scale) % 2)[..., None] == 0, 80, 100).astype(np.uint8).repeat(3, -1)
    for i, row in enumerate(rows):
        d.text((6, i * S + S // 2 - 5), labels[i][:30], fill=(235, 235, 235))
        for j, a in enumerate(row):
            tile = Image.fromarray(chk, "RGB")
            im = Image.fromarray(a, "RGBA").resize((S, S), Image.Resampling.NEAREST)
            tile.paste(im, (0, 0), im)
            sheet.paste(tile, (label_w + j * S, i * S))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/pixelgen_ema.pt")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=6, help="number of seeds per prompt (0..n-1)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--guidance-sweep", default="", help="comma list; overrides --guidance")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="grid.png")
    args = ap.parse_args()

    m = SpriteModel(args.ckpt, device="cpu", threads=args.threads)
    scales = [float(g) for g in args.guidance_sweep.split(",")] if args.guidance_sweep else [args.guidance]
    for g in scales:
        rows = [[m.generate(p, seed=s, steps=args.steps, guidance=g)[0] for s in range(args.seeds)]
                for p in args.prompts]
        out = args.out if len(scales) == 1 else args.out.replace(".png", f"_g{g:g}.png")
        make_sheet(rows, args.prompts).save(out)
        print(f"saved {out}  (steps={args.steps}, guidance={g})")


if __name__ == "__main__":
    main()
