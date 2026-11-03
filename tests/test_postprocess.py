"""Tests for post-processing utilities."""
from PIL import Image

from pixelgen.postprocess import pixelate, quantise_colors
from pixelgen.encode import image_to_hex, hex_to_image


def _make_test_image(w: int = 64, h: int = 64) -> Image.Image:
    """Create a simple gradient test image."""
    img = Image.new("RGBA", (w, h))
    for y in range(h):
        for x in range(w):
            r = int(255 * x / w)
            g = int(255 * y / h)
            img.putpixel((x, y), (r, g, 128, 255))
    return img


def test_pixelate_preserves_size():
    img = _make_test_image(512, 512)
    result = pixelate(img, pixel_size=8)
    assert result.size == (512, 512)


def test_pixelate_creates_uniform_blocks():
    """After pixelation, each 8×8 block should be a single colour."""
    img = _make_test_image(512, 512)
    result = pixelate(img, pixel_size=8)
    # Check a few blocks
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
    # All pixels should keep alpha=128
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
    # Old slow method from Studio code
    pixels = list(img.convert("RGBA").getdata())
    slow = "".join(f"{r:02X}{g:02X}{b:02X}{a:02X}" for r, g, b, a in pixels)
    assert fast == slow
