from pixelgen.server import INDEX_HTML, SpriteRequest, health


def _run(coro):
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value


def test_sprite_request_model():
    req = SpriteRequest(prompt="dragon", width=16, height=16, steps=10, guidance=2.0)
    assert req.prompt == "dragon"
    assert req.steps == 10
    assert req.guidance == 2.0
    assert req.width == 16
    assert req.height == 16


def test_index_html_template():
    assert "<!DOCTYPE html>" in INDEX_HTML
    assert "SpriteStack PixelGen" in INDEX_HTML
    assert "Generate Sprite" in INDEX_HTML
    assert "Model Variant" in INDEX_HTML
    assert "checkerboard" in INDEX_HTML
    assert "fm16" in INDEX_HTML


def test_health_endpoint():
    res = _run(health())
    assert res["status"] == "ok"
    assert "fm16" in res["variants"]
