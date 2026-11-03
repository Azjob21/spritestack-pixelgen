"""
Post-processing utilities: pixelation, resize, quantisation.

Extracted from the Studio's real_generate_sprite.py for isolated testing.
"""
from __future__ import annotations

from PIL import Image


def pixelate(img: Image.Image, pixel_size: int = 8) -> Image.Image:
    """
    Nearest-neighbour downscale then upscale to create hard pixel boundaries.

    pixel_size=8 on a 512×512 image gives a 64×64 effective grid.
    """
    w, h = img.size
    small = img.resize((w // pixel_size, h // pixel_size), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)


def quantise_colors(
    img: Image.Image,
    num_colors: int = 32,
    palette: list[tuple[int, int, int]] | None = None,
) -> Image.Image:
    """
    Reduce colour count. Preserves alpha channel.

    Parameters
    ----------
    img : Image.Image
        Input RGBA image.
    num_colors : int
        Number of colours for generic quantisation.
    palette : list of RGB tuples, optional
        If given, snap each pixel to the nearest palette colour instead of
        using generic quantisation. (Not yet implemented in v0.)
    """
    img = img.convert("RGBA")
    alpha = img.split()[3]

    rgb = img.convert("RGB")
    quantised = rgb.quantize(
        colors=num_colors,
        method=Image.Quantize.MAXCOVERAGE,
        dither=0,
    )

    result = quantised.convert("RGBA")
    result.putalpha(alpha)
    return result
