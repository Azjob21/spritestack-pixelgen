import numpy as np
from pixelgen.postprocess import drop_detached, drop_outside_grey, postprocess


def test_drop_detached_removes_far_blob_keeps_near_and_large():
    a = np.zeros((40, 40), bool)
    a[10:30, 10:30] = True      # main, 400 px
    a[5:8, 32:35] = True        # small far blob -> dropped
    a[7:9, 8:10] = True         # small blob within gap of main -> kept
    a[35:39, 1:15] = True       # 56 px < 25% of 400 -> dropped
    out = drop_detached(a, gap=2)
    assert out[15, 15] and out[7, 9] and not out[6, 33] and not out[37, 5]


def test_drop_detached_keeps_big_floating_part():
    a = np.zeros((40, 60), bool)
    a[5:35, 5:25] = True        # 600 px
    a[10:30, 40:50] = True      # 200 px detached, >= 25% -> kept
    assert drop_detached(a)[15, 45]


def test_grey_flood_eats_border_wedge_but_not_black_outline_or_colour():
    rgb = np.zeros((20, 20, 3), np.uint8)
    rgb[:] = (200, 40, 40)                      # red sprite
    rgb[:5, :5] = (110, 110, 110)               # grey wedge in the corner
    rgb[10, 10] = (10, 10, 10)                  # black pixel stays
    alpha = np.ones((20, 20), bool)
    out = drop_outside_grey(rgb, alpha)
    assert not out[0, 0] and out[10, 10] and out[15, 15]


def test_grey_flood_disabled():
    rgb = np.full((8, 8, 3), 110, np.uint8)
    alpha = np.ones((8, 8), bool)
    assert drop_outside_grey(rgb, alpha, chroma_max=0).all()


def test_postprocess_still_returns_exact_size():
    im = __import__("PIL.Image", fromlist=["Image"]).new("RGB", (512, 512), (128, 128, 128))
    assert postprocess(im, 64, 64).size == (64, 64)
