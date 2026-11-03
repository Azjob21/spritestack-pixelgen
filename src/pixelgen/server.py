"""
Studio-compatible FastAPI server.

Provides the same POST /generate-sprite endpoint that SpriteStack Studio
expects, so this repo can be used as a drop-in replacement for the Studio's
built-in generator during development.

Usage:
    python -m pixelgen.server --port 8000
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    raise ImportError(
        "Server dependencies not installed. Run: pip install -e '.[server]'"
    )

from .pipeline import generate, load_pipeline
from .encode import image_to_hex

app = FastAPI(title="SpriteStack PixelGen", version="0.1.0")
_executor = ThreadPoolExecutor(max_workers=1)


@app.on_event("startup")
async def warmup():
    """Pre-load the pipeline so the first request doesn't pay cold-start."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, load_pipeline)


@app.get("/health")
async def health():
    return {"status": "ok", "model": "spritestack-pixelgen-v0"}


@app.post("/generate-sprite")
async def generate_sprite(data: dict):
    """Studio-compatible sprite generation endpoint."""
    prompt = (data.get("prompt") or "sprite").strip()
    width = max(1, int(data.get("width") or 32))
    height = max(1, int(data.get("height") or 32))
    seed = data.get("seed")  # None = use pipeline default

    loop = asyncio.get_event_loop()

    # Run inference in a thread pool to avoid blocking the event loop
    # (fixes ISSUE(v0) #5: blocking inference)
    img = await loop.run_in_executor(
        _executor,
        partial(generate, prompt=prompt, width=width, height=height, seed=seed),
    )

    hex_str = image_to_hex(img)

    return JSONResponse({
        "hex": hex_str,
        "width": width,
        "height": height,
        "prompt": prompt,
        "model": "spritestack-pixelgen-v0",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
