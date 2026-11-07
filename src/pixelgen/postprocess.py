"""
Post-processing: background keying, grid snapping, palette mapping, cleanup.

New path (``postprocess``)
--------------------------
1. ``key_background``  - border-connected background mask at full resolution,
                         background colour estimated from the image edges.
2. ``to_grid``         - each output pixel = per-channel median of the
                         *foreground* samples in its cell (no background bleed,
                         no NEAREST sampling noise, no skipped cells).
3. ``clean_alpha`` / ``fill_pinholes`` - drop stray specks, fix 1px holes.
4. ``snap_to_palette`` (CIELAB) or ``quantise_opaque`` - palette discipline.
5. optional 1px outline.

The v0 chain (``legacy_postprocess``) is kept verbatim so the baseline stays
reproducible.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .palettes import PALETTES

RGB = tuple[int, int, int]
PaletteArg = Union[None, str, Sequence[RGB]]

_BOX = Image.Resampling.BOX


# --------------------------------------------------------------------------- #
# Legacy (v0) chain - kept so the baseline is reproducible
# --------------------------------------------------------------------------- #
def pixelate(img: Image.Image, pixel_size: int = 8) -> Image.Image:
    """Nearest-neighbour down/up scale (v0)."""
    w, h = img.size
    small = img.resize((w // pixel_size, h // pixel_size), Image.Resampling.NEAREST)
    return small.resize((w, h), Image.Resampling.NEAREST)


def quantise_colors(img: Image.Image, num_colors: int = 32) -> Image.Image:
    """Generic PIL quantisation preserving alpha (v0)."""
    img = img.convert("RGBA")
    alpha = img.split()[3]
    q = img.convert("RGB").quantize(
        colors=num_colors, method=Image.Quantize.MAXCOVERAGE, dither=0
    )
    out = q.convert("RGBA")
    out.putalpha(alpha)
    return out


def legacy_postprocess(image: Image.Image, width: int, height: int,
                       num_colors: int = 32) -> Image.Image:
    """Exact v0 behaviour: pixelate(8) -> NEAREST resize -> quantise."""
    image = pixelate(image.convert("RGBA"), pixel_size=8)
    if width <= 32:
        image = image.resize((256, 256), Image.Resampling.NEAREST)
    image = image.resize((width, height), Image.Resampling.NEAREST)
    return quantise_colors(image, num_colors=num_colors)


# --------------------------------------------------------------------------- #
# Colour helpers
# --------------------------------------------------------------------------- #
_M = np.array([[0.4124564, 0.3575761, 0.1804375],
               [0.2126729, 0.7151522, 0.0721750],
               [0.0193339, 0.1191920, 0.9503041]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 (..., 3) -> CIELAB (..., 3), D65. Pure numpy."""
    c = np.asarray(rgb, np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _M.T) / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def resolve_palette(palette: PaletteArg) -> Optional[list[RGB]]:
    """Accept None, a registry name ('db32'), or an explicit RGB list."""
    if palette is None:
        return None
    if isinstance(palette, str):
        key = palette.lower().replace("-", "").replace("_", "")
        if key not in PALETTES:
            raise ValueError(
                f"Unknown palette {palette!r}. Available: {sorted(PALETTES)}"
            )
        return list(PALETTES[key])
    return [tuple(int(v) for v in c) for c in palette]  # type: ignore[misc]


def snap_to_palette(rgb: np.ndarray, palette: Sequence[RGB]) -> np.ndarray:
    """Map every pixel to its nearest palette colour in CIELAB."""
    pal = np.asarray(palette, np.uint8)
    flat = rgb.reshape(-1, 3)
    d = ((srgb_to_lab(flat)[:, None, :] - srgb_to_lab(pal)[None, :, :]) ** 2).sum(-1)
    return pal[d.argmin(axis=1)].reshape(rgb.shape)


def quantise_opaque(rgb: np.ndarray, alpha: np.ndarray, num_colors: int) -> np.ndarray:
    """Generic quantisation over *opaque pixels only* (key colour never takes a slot)."""
    out = rgb.copy()
    px = rgb[alpha]
    if len(px) == 0 or len(np.unique(px, axis=0)) <= num_colors:
        return out
    im = Image.fromarray(np.ascontiguousarray(px.reshape(1, -1, 3)))
    q = im.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT,
                    dither=Image.Dither.NONE)
    out[alpha] = np.asarray(q.convert("RGB")).reshape(-1, 3)
    return out


# --------------------------------------------------------------------------- #
# Background keying
# --------------------------------------------------------------------------- #
def _border_labels(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(np.concatenate(
        [labels[0], labels[-1], labels[1:-1, 0], labels[1:-1, -1]]))
    return ids[ids != 0]


def _grad_mag(f: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Per-pixel colour gradient magnitude (summed over channels) of a lightly blurred image."""
    sm = ndi.gaussian_filter(f, sigma=(sigma, sigma, 0))
    gy, gx = np.gradient(sm, axis=(0, 1))
    return np.sqrt((gx ** 2 + gy ** 2).sum(-1))


def key_background(
    rgb: np.ndarray,
    tol: float = 60.0,
    grow_px: int = 2,
    enclosed_min_frac: float = 0.004,
    min_border_frac: float = 0.5,
    edge_thr: Optional[float] = 2.0,
    far_tol: float = 110.0,
) -> np.ndarray:
    """
    Boolean mask (True = background) for an image on a flat or gently shaded backdrop.

    Two cooperating detectors, combined with OR:

    1. *Colour keying*: pixels within ``tol`` of the backdrop colour (median of the
       four corner patches) that connect to the border, plus large enclosed pockets
       (>= ``enclosed_min_frac`` of the image, e.g. between legs).
    2. *Edge-bounded flood* (``edge_thr`` not None): flood from the border through
       *smooth* pixels (gradient <= ``edge_thr`` per pixel) that are at most
       ``far_tol`` from the backdrop colour. Sprite outlines are sharp edges and stop
       the flood; vignettes, lighting gradients and soft halos are smooth and are
       swallowed. ``far_tol`` caps how different from the backdrop a pixel may be,
       so a leak through a gap in an outline cannot eat a strongly coloured sprite.
       Set ``edge_thr=None`` (or <= 0) for the pure colour-key behaviour. Lower values
       treat softer transitions as edges and keep more low-contrast detail; higher
       values remove more halo and gradient but can eat soft grey-on-grey objects.

    ``grow_px`` then eats anti-aliased fringe: the mask expands by up to that many
    pixels into pixels within a looser tolerance of the backdrop. If fewer than
    ``min_border_frac`` of border pixels look like the backdrop (busy backdrop,
    sprite filling the frame) nothing is removed rather than guessing.
    """
    f = np.asarray(rgb, np.float32)
    if edge_thr is not None and edge_thr <= 0:
        edge_thr = None
    border = np.concatenate([f[0], f[-1], f[1:-1, 0], f[1:-1, -1]])
    p = max(2, min(f.shape[:2]) // 64)
    corners = np.concatenate([c.reshape(-1, 3) for c in
                              (f[:p, :p], f[:p, -p:], f[-p:, :p], f[-p:, -p:])])
    bg = np.median(corners, axis=0)
    gate = tol if edge_thr is None else max(tol, far_tol)
    if (np.linalg.norm(border - bg, axis=1) <= gate).mean() < min_border_frac:
        return np.zeros(f.shape[:2], bool)

    dist = np.linalg.norm(f - bg, axis=2)
    cand = dist <= tol
    labels, n = ndi.label(cand)
    if n == 0 and edge_thr is None:
        return np.zeros(f.shape[:2], bool)

    mask = np.isin(labels, _border_labels(labels)) if n else np.zeros(f.shape[:2], bool)
    if n and enclosed_min_frac and enclosed_min_frac > 0:
        sizes = ndi.sum(cand, labels, index=np.arange(1, n + 1))
        big = np.flatnonzero(sizes >= enclosed_min_frac * cand.size) + 1
        mask |= np.isin(labels, big)

    loose = dist <= tol * 1.8
    if edge_thr is not None:
        edge = ndi.binary_dilation(_grad_mag(f) > edge_thr, iterations=1)
        smooth = cand | (~edge & (dist <= far_tol))
        sl, sn = ndi.label(smooth)
        if sn:
            mask |= np.isin(sl, _border_labels(sl))
        loose = dist <= max(tol * 1.8, far_tol)

    if grow_px > 0:
        mask = mask | ndi.binary_dilation(mask, iterations=grow_px, mask=loose)
    return mask


# --------------------------------------------------------------------------- #
# Grid snapping
# --------------------------------------------------------------------------- #
def _box(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(a.astype(np.float32)).resize(size, _BOX))


def to_grid(
    rgb: np.ndarray,
    bg_mask: np.ndarray,
    out_w: int,
    out_h: int,
    cell: int = 4,
    min_cover: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Downscale to the target grid.

    The image is box-averaged to ``cell`` x ``cell`` samples per output pixel
    using only foreground pixels (so background colour never bleeds in), then
    each output pixel takes the per-channel median of its foreground samples.
    An output pixel is opaque if at least ``min_cover`` of its samples are
    foreground.

    Returns (rgb uint8 [h, w, 3], alpha bool [h, w]).
    """
    k = max(1, int(cell))
    size = (out_w * k, out_h * k)
    fg = (~bg_mask).astype(np.float32)
    weight = _box(fg, size)
    chans = [
        _box(rgb[..., c].astype(np.float32) * fg, size) / np.maximum(weight, 1e-6)
        for c in range(3)
    ]
    samples = np.stack(chans, axis=-1)
    is_fg = weight >= 0.5

    def cells(a: np.ndarray) -> np.ndarray:
        extra = a.shape[2:]
        a = a.reshape(out_h, k, out_w, k, *extra)
        a = a.transpose(0, 2, 1, 3, *range(4, 4 + len(extra)))
        return a.reshape(out_h, out_w, k * k, *extra)

    cs, cf = cells(samples), cells(is_fg)
    cover = cf.mean(axis=-1)
    masked = np.where(cf[..., None], cs, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(masked, axis=2)
    med = np.clip(np.rint(np.nan_to_num(med, nan=0.0)), 0, 255).astype(np.uint8)
    return med, cover >= min_cover


# --------------------------------------------------------------------------- #
# Cleanup
# --------------------------------------------------------------------------- #
def clean_alpha(alpha: np.ndarray, min_rel_area: float = 0.05, min_abs: int = 2) -> np.ndarray:
    """
    Drop disconnected specks. Components (8-connected) smaller than
    ``max(min_abs, min_rel_area * largest)`` are removed; detached but
    substantial parts (a floating sword) survive. Unlike a colour-neighbour
    rule this never deletes 1px details *inside* the sprite.
    """
    lab, n = ndi.label(alpha, structure=np.ones((3, 3), int))
    if n == 0:
        return alpha
    sizes = ndi.sum(alpha, lab, index=np.arange(1, n + 1))
    keep = sizes >= max(min_abs, min_rel_area * sizes.max())
    return np.isin(lab, np.flatnonzero(keep) + 1)


def drop_detached(alpha: np.ndarray, gap: int = 2, keep_rel_area: float = 0.25) -> np.ndarray:
    """
    Remove stray objects: keep the largest component, plus any component that lies
    within ``gap`` grid pixels of it, plus any detached one at least ``keep_rel_area``
    of its size (a floating weapon). Smaller detached blobs (stray props, shadow
    slabs) are dropped.
    """
    lab, n = ndi.label(alpha, structure=np.ones((3, 3), int))
    if n <= 1:
        return alpha
    sizes = ndi.sum(alpha, lab, index=np.arange(1, n + 1))
    main = int(sizes.argmax()) + 1
    near = ndi.binary_dilation(lab == main, structure=np.ones((3, 3), bool), iterations=max(1, gap))
    keep = np.zeros(n + 1, bool)
    for i in range(1, n + 1):
        keep[i] = i == main or sizes[i - 1] >= keep_rel_area * sizes[main - 1] or near[lab == i].any()
    return keep[lab]


def drop_outside_grey(rgb: np.ndarray, alpha: np.ndarray, chroma_max: float = 8.0,
                      l_min: float = 25.0, l_max: float = 85.0) -> np.ndarray:
    """
    Second keying pass on the output grid: flood inward from the transparent area and
    the image border through opaque *neutral grey* pixels (Lab chroma <= ``chroma_max``,
    lightness within [l_min, l_max]). Removes leftover backdrop wedges and halos while
    black outlines (L < l_min) and white highlights (L > l_max) stop the flood.
    ``chroma_max <= 0`` disables it.
    """
    if chroma_max <= 0 or not alpha.any():
        return alpha
    lab = srgb_to_lab(rgb)
    grey = alpha & (np.hypot(lab[..., 1], lab[..., 2]) <= chroma_max) \
        & (lab[..., 0] >= l_min) & (lab[..., 0] <= l_max)
    if not grey.any():
        return alpha
    lab_c, n = ndi.label(grey, structure=np.ones((3, 3), int))
    outside = ~alpha
    touch = ndi.binary_dilation(outside, structure=np.ones((3, 3), bool))
    edge = np.zeros_like(alpha)
    edge[0], edge[-1], edge[:, 0], edge[:, -1] = True, True, True, True
    ids = np.unique(lab_c[(touch | edge) & grey])
    return alpha & ~np.isin(lab_c, ids[ids != 0])


def fill_pinholes(rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill single-pixel transparent holes surrounded by sprite with the nearest colour."""
    holes = ndi.binary_fill_holes(alpha) & ~alpha
    if not holes.any():
        return rgb, alpha
    lab, n = ndi.label(holes)
    sizes = ndi.sum(holes, lab, index=np.arange(1, n + 1))
    tiny = np.isin(lab, np.flatnonzero(sizes <= 1) + 1)
    if not tiny.any():
        return rgb, alpha
    _, (iy, ix) = ndi.distance_transform_edt(~alpha, return_indices=True)
    rgb = rgb.copy()
    rgb[tiny] = rgb[iy[tiny], ix[tiny]]
    return rgb, alpha | tiny


def add_outline(rgb: np.ndarray, alpha: np.ndarray, colour: RGB) -> tuple[np.ndarray, np.ndarray]:
    """1px 4-connected outline on transparent pixels touching the sprite."""
    ring = ndi.binary_dilation(alpha) & ~alpha
    rgb = rgb.copy()
    rgb[ring] = colour
    return rgb, alpha | ring


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def postprocess(
    image: Image.Image,
    width: int,
    height: int,
    *,
    palette: PaletteArg = None,
    num_colors: int = 32,
    transparent: bool = True,
    cleanup: bool = True,
    outline: bool = False,
    outline_color: RGB = (24, 20, 37),
    grey_chroma: float = 8.0,
    detach_gap: Optional[int] = 2,
    bg_tol: float = 60.0,
    edge_thr: Optional[float] = 2.0,
    cell: int = 4,
) -> Image.Image:
    """Full new-style chain. Returns an RGBA image of exactly (width, height)."""
    rgb = np.asarray(image.convert("RGB"), np.uint8)
    bg = key_background(rgb, tol=bg_tol, edge_thr=edge_thr) if transparent else np.zeros(rgb.shape[:2], bool)

    grid, alpha = to_grid(rgb, bg, width, height, cell=cell)
    if cleanup:
        alpha = clean_alpha(alpha)
        alpha = drop_outside_grey(grid, alpha, chroma_max=grey_chroma)
        if detach_gap is not None:
            alpha = drop_detached(alpha, gap=detach_gap)
        grid, alpha = fill_pinholes(grid, alpha)

    pal = resolve_palette(palette)
    grid = snap_to_palette(grid, pal) if pal else quantise_opaque(grid, alpha, num_colors)

    if outline:
        grid, alpha = add_outline(grid, alpha, outline_color)

    grid = grid.copy()
    grid[~alpha] = 0
    out = np.dstack([grid, alpha.astype(np.uint8) * 255])
    return Image.fromarray(out)
