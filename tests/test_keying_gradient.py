import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from pixelgen.postprocess import key_background, postprocess

RED = (200, 30, 30)


def gradient_backdrop(size=512, lo=105, hi=160, seed=0):
    """Horizontal+vertical lighting gradient with mild noise, like the raw SD backdrops."""
    yy, xx = np.mgrid[0:size, 0:size] / (size - 1)
    base = hi - (hi - lo) * (0.6 * (1 - xx) * yy + 0.4 * yy * (1 - xx))   # dark bottom-left
    rng = np.random.default_rng(seed)
    base = base + rng.normal(0, 0.8, base.shape)
    return np.clip(np.dstack([base, base, base]), 0, 255).astype(np.uint8)


def test_gradient_backdrop_fully_keyed_where_colour_key_fails():
    a = gradient_backdrop()
    a[150:400, 200:320] = RED
    old = key_background(a, tol=25, edge_thr=None)
    new = key_background(a, tol=25)
    assert not old[480, 20]                    # dark corner survives the pure colour key
    assert new[480, 20] and new[10, 10] and new[500, 500]
    assert not new[250, 250]                   # sprite untouched


def test_soft_glow_halo_is_removed():
    a = np.full((512, 512, 3), 150, np.uint8)
    glow = np.zeros((512, 512), np.float32)
    glow[180:340, 180:340] = 1.0
    glow = ndi.gaussian_filter(glow, 25) * 30            # +30 levels, very soft
    a = np.clip(a + glow[..., None], 0, 255).astype(np.uint8)
    a[200:320, 200:320] = RED
    m = key_background(a, tol=25)
    assert m[190, 190] and m[330, 260] and not m[260, 260]


def test_sharp_outline_stops_the_flood_small_holes_stay():
    a = np.full((512, 512, 3), 150, np.uint8)
    a[100:140, 100:140] = RED                            # ring: 40px block ...
    a[112:128, 112:128] = 150                            # ... with a 16px backdrop-coloured hole
    m = key_background(a, tol=25)
    assert m[0, 0] and not m[120, 120]                   # hole (0.1% of image) is not swallowed


def test_strongly_coloured_sprite_never_eaten_even_with_soft_edges():
    a = gradient_backdrop()
    blob = np.zeros((512, 512), np.float32); blob[200:320, 200:320] = 1
    blob = ndi.gaussian_filter(blob, 6)[..., None]
    a = (a * (1 - blob) + np.array(RED) * blob).astype(np.uint8)
    m = key_background(a, tol=25)
    assert not m[260, 260] and m[5, 5]


def test_end_to_end_gradient_backdrop_gives_clean_transparency():
    a = gradient_backdrop()
    a[150:400, 200:320] = RED
    out = np.asarray(postprocess(Image.fromarray(a), 32, 32, bg_tol=25))
    assert (out[..., 3][0, :] == 0).all() and (out[..., 3][:, 0] == 0).all()
    assert out[16, 16, 3] == 255


def test_edge_thr_zero_means_pure_colour_key():
    a = gradient_backdrop(); a[150:400, 200:320] = RED
    assert (key_background(a, tol=25, edge_thr=0) == key_background(a, tol=25, edge_thr=None)).all()
