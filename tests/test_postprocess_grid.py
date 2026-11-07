import numpy as np
import pytest
from PIL import Image

from pixelgen.palettes import DB32, PALETTES
from pixelgen.postprocess import (
    add_outline, clean_alpha, fill_pinholes, key_background, postprocess,
    resolve_palette, snap_to_palette, srgb_to_lab, to_grid,
)

BG = (250, 20, 240)       # "SD-ish" magenta, not exact #FF00FF
RED, BLUE, YELLOW = (200, 30, 30), (30, 60, 200), (240, 220, 40)


def canvas(size=512, noise=0, seed=0):
    a = np.zeros((size, size, 3), np.uint8)
    a[:] = BG
    if noise:
        rng = np.random.default_rng(seed)
        a = np.clip(a + rng.integers(-noise, noise + 1, a.shape), 0, 255).astype(np.uint8)
    return a


def rect(a, x0, y0, x1, y1, col):
    a[y0:y1, x0:x1] = col


def test_corners_transparent_centre_opaque_and_no_key_colour():
    a = canvas(noise=12)
    rect(a, 128, 128, 384, 384, RED)
    out = np.asarray(postprocess(Image.fromarray(a), 32, 32))
    assert out.shape == (32, 32, 4)
    assert set(np.unique(out[..., 3])) == {0, 255}
    assert out[0, 0, 3] == 0 and out[-1, -1, 3] == 0
    assert out[16, 16, 3] == 255
    opaque = out[out[..., 3] == 255][:, :3]
    assert (opaque.astype(int) - np.array(RED)).__abs__().max() <= 3   # no magenta bleed


def test_antialiased_edge_leaves_no_pink_fringe():
    a = canvas()
    rect(a, 128, 128, 384, 384, RED)
    # 2px blended ring around the sprite (50% sprite / 50% bg)
    blend = ((np.array(RED) + np.array(BG)) / 2).astype(np.uint8)
    a[126:128, 126:386] = blend; a[384:386, 126:386] = blend
    a[126:386, 126:128] = blend; a[126:386, 384:386] = blend
    out = np.asarray(postprocess(Image.fromarray(a), 32, 32))
    opaque = out[out[..., 3] == 255][:, :3].astype(int)
    assert np.abs(opaque - np.array(RED)).max() <= 3


def test_one_pixel_detail_survives_at_32():
    a = canvas()
    rect(a, 128, 128, 384, 384, RED)
    rect(a, 224, 224, 240, 240, YELLOW)          # exactly one 16px output cell
    out = np.asarray(postprocess(Image.fromarray(a), 32, 32, num_colors=8))
    yx = (224 // 16, 224 // 16)
    assert out[yx][3] == 255
    assert out[yx][0] > 200 and out[yx][2] < 100  # still yellow-ish, not eaten by neighbours


def test_median_beats_speckle():
    a = canvas()
    rect(a, 128, 128, 384, 384, RED)
    rng = np.random.default_rng(1)
    ys, xs = rng.integers(130, 380, 400), rng.integers(130, 380, 400)
    a[ys, xs] = BLUE                              # ~0.6% salt noise inside the sprite
    out = np.asarray(postprocess(Image.fromarray(a), 32, 32, num_colors=8))
    inner = out[10:22, 10:22, :3].reshape(-1, 3).astype(int)
    assert np.abs(inner - np.array(RED)).max() <= 3


def test_enclosed_pocket_removed_but_small_enclosed_kept():
    a = canvas()
    rect(a, 64, 64, 448, 448, RED)
    rect(a, 200, 200, 300, 300, BG)               # big pocket (~3.8% of image) -> transparent
    rect(a, 380, 380, 388, 388, BG)               # tiny pocket (< threshold) -> stays
    m = key_background(a)
    assert m[250, 250] and not m[384, 384]


def test_busy_backdrop_falls_back_to_no_keying():
    rng = np.random.default_rng(3)
    a = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    assert not key_background(a).any()


def test_sprite_touching_edge_still_keyed_from_corners():
    a = canvas()
    rect(a, 100, 100, 412, 512, RED)              # runs off the bottom edge
    m = key_background(a)
    assert m[10, 10] and m[500, 20] and not m[300, 250]


def test_non_divisor_target_size():
    a = canvas()
    rect(a, 100, 100, 400, 400, RED)
    out = postprocess(Image.fromarray(a), 48, 40)
    assert out.size == (48, 40)


def test_palette_membership_and_registry():
    a = canvas()
    rect(a, 100, 100, 400, 400, (150, 80, 60))
    out = np.asarray(postprocess(Image.fromarray(a), 32, 32, palette="db32"))
    cols = {tuple(c) for c in out[out[..., 3] == 255][:, :3]}
    assert cols <= set(DB32)
    assert resolve_palette("PICO-8") == PALETTES["pico8"]
    with pytest.raises(ValueError):
        resolve_palette("nope")


def test_palette_snap_is_identity_on_palette_colours_and_lab_sane():
    arr = np.array(DB32, np.uint8).reshape(1, -1, 3)
    assert (snap_to_palette(arr, DB32) == arr).all()
    lab = srgb_to_lab(np.array([[255, 255, 255], [0, 0, 0]]))
    assert abs(lab[0, 0] - 100) < 0.5 and abs(lab[1, 0]) < 0.5


def test_clean_alpha_removes_specks_keeps_inner_detail_and_big_parts():
    al = np.zeros((32, 32), bool)
    al[8:24, 8:24] = True            # main body
    al[2, 2] = True                  # speck
    al[2:7, 28:31] = True            # 15px detached part (>5% of 256) -> kept
    out = clean_alpha(al)
    assert not out[2, 2] and out[3, 29] and out[16, 16]


def test_fill_pinholes_only_fills_single_pixels():
    rgb = np.zeros((16, 16, 3), np.uint8); rgb[:] = RED
    al = np.zeros((16, 16), bool); al[2:14, 2:14] = True
    al[5, 5] = False                  # 1px hole
    al[8:10, 8:10] = False            # 2x2 hole - intentional gap, keep
    rgb2, al2 = fill_pinholes(rgb, al)
    assert al2[5, 5] and not al2[8, 8]
    assert tuple(rgb2[5, 5]) == RED


def test_outline_adds_ring_outside_only():
    rgb = np.zeros((8, 8, 3), np.uint8); al = np.zeros((8, 8), bool); al[3:5, 3:5] = True
    rgb2, al2 = add_outline(rgb, al, (1, 2, 3))
    assert al2.sum() == 4 + 8 and tuple(rgb2[2, 3]) == (1, 2, 3) and not al2[2, 2]


def test_transparent_false_keeps_everything_opaque():
    a = canvas(); rect(a, 100, 100, 400, 400, RED)
    out = np.asarray(postprocess(Image.fromarray(a), 32, 32, transparent=False))
    assert (out[..., 3] == 255).all()


def test_to_grid_shapes():
    rgb = np.zeros((512, 512, 3), np.uint8)
    g, al = to_grid(rgb, np.zeros((512, 512), bool), 64, 64)
    assert g.shape == (64, 64, 3) and al.shape == (64, 64) and al.all()


def test_speed_budget_512_to_64():
    import time
    a = canvas(noise=10); rect(a, 100, 100, 400, 400, RED)
    img = Image.fromarray(a)
    t = time.perf_counter(); postprocess(img, 64, 64); dt = time.perf_counter() - t
    assert dt < 1.0, f"post-processing too slow: {dt:.2f}s"
