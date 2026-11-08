import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from pixelgen.pipeline import generate

ITEMS = [
    ("red dragon", 13, "Red Dragon"),
    ("wizard with a staff", 21, "Wizard"),
    ("small green slime", 42, "Green Slime"),
    ("purple bat", 7, "Purple Bat"),
    ("golden treasure chest", 5, "Treasure Chest"),
    ("steel sword with blue gem", 1, "Steel Sword"),
    ("red health potion", 3, "Health Potion"),
    ("knight in silver armor", 10, "Knight"),
    ("goblin warrior", 4, "Goblin"),
    ("magic glowing staff", 8, "Magic Staff"),
    ("wooden round shield", 6, "Wooden Shield"),
    ("black cat", 2, "Black Cat"),
]

# Generate raw 16x16 sprites
sprites = []
for prompt, seed, label in ITEMS:
    res = generate(prompt, 16, 16, seed=seed, config="fm16")
    sprites.append((res.image, label, prompt))
    print(f"Generated {label} in {res.timings['total_s']:.2f}s")

# Build a composite high-resolution showcase sheet
# 6 columns x 2 rows
COLS = 6
ROWS = (len(sprites) + COLS - 1) // COLS
SCALE = 6        # 16 * 6 = 96px sprite
PAD = 16
CELL_W = 16 * SCALE + PAD * 2   # 96 + 32 = 128
CELL_H = 16 * SCALE + 48        # 96 + 48 = 144
HEADER_H = 56

TOTAL_W = COLS * CELL_W + PAD * 2
TOTAL_H = ROWS * CELL_H + HEADER_H + PAD * 2

# Background canvas
canvas = Image.new("RGBA", (TOTAL_W, TOTAL_H), (17, 19, 26, 255))
draw = ImageDraw.Draw(canvas)

# Title & Subtitle
draw.text((PAD + 12, 12), "SpriteStack PixelGen — Native 16×16 Sprite Sheet (fm16)", fill=(240, 245, 255))
draw.text((PAD + 12, 32), "True 16×16 native resolution · ~0.78s CPU inference · Direct learned RGBA transparency", fill=(140, 160, 180))

# Checkerboard tile generator for transparent sprite backdrop
S = 16 * SCALE
yy, xx = np.mgrid[0:S, 0:S]
chk = np.where(((yy // SCALE + xx // SCALE) % 2)[..., None] == 0, 28, 38).astype(np.uint8).repeat(3, -1)
chk_tile = Image.fromarray(chk, "RGB")

for idx, (img, label, prompt) in enumerate(sprites):
    c = idx % COLS
    r = idx // COLS
    x0 = PAD * 2 + c * CELL_W
    y0 = HEADER_H + PAD + r * CELL_H

    # Cell background card
    draw.rounded_rectangle([x0, y0, x0 + CELL_W - PAD, y0 + CELL_H - 10], radius=6, fill=(24, 27, 36), outline=(42, 47, 62), width=1)

    # Paste checkerboard backdrop
    sprite_x = x0 + (CELL_W - PAD - S) // 2
    sprite_y = y0 + 10
    canvas.paste(chk_tile, (sprite_x, sprite_y))

    # Scale sprite nearest-neighbour
    scaled_sprite = img.resize((S, S), Image.Resampling.NEAREST)
    canvas.paste(scaled_sprite, (sprite_x, sprite_y), scaled_sprite)

    # Label text centered
    text_bbox = draw.textbbox((0, 0), label)
    text_w = text_bbox[2] - text_bbox[0]
    draw.text((x0 + (CELL_W - PAD - text_w) // 2, sprite_y + S + 6), label, fill=(225, 230, 240))

# Save outputs
out1 = Path("docs/samples/fm16_showcase_sheet.png")
canvas.save(out1)
print(f"Saved {out1} ({canvas.size[0]}x{canvas.size[1]})")

main_out = Path("C:/Users/Administrateur/Desktop/spritestack/docs/screenshots/fm16_showcase_sheet.png")
canvas.save(main_out)
print(f"Saved {main_out} ({canvas.size[0]}x{canvas.size[1]})")
