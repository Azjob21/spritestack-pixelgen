"""
RGBA encoding utilities.

Provides the hex-stream format expected by the SpriteStack Studio frontend.
"""
from __future__ import annotations

from PIL import Image


def image_to_hex(img: Image.Image) -> str:
    """
    Encode a PIL RGBA image to an uppercase hex string: RRGGBBAARRGGBBAA...

    Uses img.tobytes().hex() instead of a per-pixel loop (>10× faster).
    """
    img = img.convert("RGBA")
    return img.tobytes().hex().upper()


def hex_to_image(hex_str: str, width: int, height: int) -> Image.Image:
    """
    Decode an RRGGBBAA hex string back to a PIL RGBA image.
    """
    raw = bytes.fromhex(hex_str)
    expected = width * height * 4
    if len(raw) != expected:
        raise ValueError(
            f"Hex data has {len(raw)} bytes, expected {expected} "
            f"for {width}×{height} RGBA"
        )
    return Image.frombytes("RGBA", (width, height), raw)
