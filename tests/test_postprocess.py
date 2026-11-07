"""Tests for post-processing utilities (legacy v0 helpers + new keying/cleanup API)."""
import numpy as np
from PIL import Image

from pixelgen.postprocess import (
    add_outline, clean_alpha, fill_pinholes, key_background,
    pixelate, quantise_colors,
)
from pixelgen.encode import image_to_hex, hex_to_image

MAGENTA, RED = (255, 0, 255), (255, 0, 0)


def _make_test_image(w: int = 64, h: int = 64) -> Image.Image:
    """Create a simple gradient test image."""
    img = Image.new("RGBA", (w, h))
    for y in range(h):
        for x in range(w):
            r = int(255 * x / w)
            g = int(255 * y / h)
            img.putpixel((x, y), (r, g, 128, 255))
    return img


def _solid(w, h, colour):
    a = np.zeros((h, w, 3), np.uint8)
    a[:] = colour
    return a


# --- legacy helpers (unchanged) ----------------------------------------------
def test_pixelate_preserves_size():
    img = _make_test_image(512, 512)
    result = pixelate(img, pixel_size=8)
    assert result.size == (512, 512)


def test_pixelate_creates_uniform_blocks():
    """After pixelation, each 8×8 block should be a single colour."""
    img = _make_test_image(512, 512)
    result = pixelate(img, pixel_size=8)
    for bx in range(0, 512, 8):
        for by in range(0, 512, 8):
            block_color = result.getpixel((bx, by))
            for dx in range(8):
                for dy in range(8):
                    assert result.getpixel((bx + dx, by + dy)) == block_color


def test_quantise_reduces_colors():
    img = _make_test_image(64, 64)
    result = quantise_colors(img, num_colors=8)
    colors = set(result.getdata())
    assert len(colors) <= 8


def test_quantise_preserves_alpha():
    img = Image.new("RGBA", (16, 16), (255, 0, 0, 128))
    result = quantise_colors(img, num_colors=4)
    for pixel in result.getdata():
        assert pixel[3] == 128


def test_hex_roundtrip():
    img = _make_test_image(16, 16)
    hex_str = image_to_hex(img)
    decoded = hex_to_image(hex_str, 16, 16)
    assert list(img.getdata()) == list(decoded.getdata())


def test_image_to_hex_uses_tobytes():
    """Verify the fast path produces the same result as the old per-pixel loop."""
    img = _make_test_image(16, 16)
    fast = image_to_hex(img)
    pixels = list(img.convert("RGBA").getdata())
    slow = "".join(f"{r:02X}{g:02X}{b:02X}{a:02X}" for r, g, b, a in pixels)
    assert fast == slow


# --- background keying (replaces extract_chroma_key_alpha tests) ---------------
def test_key_background_removes_backdrop():
    a = _solid(32, 32, MAGENTA)
    a[11:21, 11:21] = RED
    m = key_background(a)
    assert m[0, 0] and m[31, 31]
    assert not m[15, 15]


def test_key_background_estimates_colour_and_respects_tolerance():
    # Off-magenta, noisy backdrop: no exact hex needed, tolerance controls strictness.
    rng = np.random.default_rng(0)
    a = _solid(64, 64, (250, 5, 250)).astype(int) + rng.integers(-15, 16, (64, 64, 3))
    a = np.clip(a, 0, 255).astype(np.uint8)
    a[24:40, 24:40] = (0, 0, 255)
    m = key_background(a, tol=60)
    assert m[0, 0] and not m[32, 32]
    # A tolerance far below the noise floor can't trust the border -> removes nothing.
    assert not key_background(a, tol=2, edge_thr=None).any()


def test_key_background_no_flood_through_sprite():
    a = _solid(16, 16, MAGENTA)
    a[8, :] = RED                                  # solid wall across the image
    m = key_background(a)
    assert m[2, 5] and m[12, 5]                    # backdrop above and below removed
    assert not m[8, :].any()                       # wall stays


def test_key_background_preserves_non_background():
    a = _solid(16, 16, MAGENTA)
    scattered = [(3, 3), (10, 7), (4, 12)]         # (y, x)
    for y, x in scattered:
        a[y, x] = (0, 255, 0)
    m = key_background(a)
    for y, x in scattered:
        assert not m[y, x]


# --- cleanup (replaces cleanup_sprite tests) ----------------------------------
def test_clean_alpha_removes_orphan_pixels():
    al = np.zeros((16, 16), bool)
    al[2:6, 2:6] = True
    al[12, 12] = True                              # isolated speck
    out = clean_alpha(al)
    assert not out[12, 12]
    assert out[2:6, 2:6].all()


def test_fill_pinholes_fills_single_pixel_hole():
    rgb = _solid(16, 16, (0, 0, 0))
    al = np.zeros((16, 16), bool)
    al[5:10, 5:10] = True
    rgb[5:10, 5:10] = RED
    al[7, 7] = False                               # pinhole
    rgb2, al2 = fill_pinholes(rgb, al)
    assert al2[7, 7]
    assert tuple(rgb2[7, 7]) == RED


def test_add_outline_draws_ring_only_next_to_sprite():
    rgb = _solid(16, 16, (0, 0, 0))
    al = np.zeros((16, 16), bool)
    al[6:10, 6:10] = True
    rgb[6:10, 6:10] = RED
    rgb2, al2 = add_outline(rgb, al, (1, 2, 3))
    assert al2[7, 5] and tuple(rgb2[7, 5]) == (1, 2, 3)
    assert not al2[0, 0]


def test_cleanup_is_noop_on_clean_sprite():
    rgb = _solid(8, 8, (0, 0, 0))
    al = np.zeros((8, 8), bool)
    al[2:6, 2:6] = True
    rgb[2:6, 2:6] = RED
    al2 = clean_alpha(al)
    rgb2, al3 = fill_pinholes(rgb, al2)
    assert (al3 == al).all() and (rgb2 == rgb).all()
