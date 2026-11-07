"""
Studio-compatible FastAPI server.

    PIXELGEN_VARIANT=s6 python -m pixelgen.server --port 8000

POST /generate-sprite  {prompt, width, height, seed?, palette?, variant?}
  -> {hex, width, height, prompt, model, seed, variant, palette, timings}
"""
from __future__ import annotations

import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    raise ImportError("Server dependencies not installed. Run: pip install -e '.[server]'")

from . import __version__
from .encode import image_to_hex
from .palettes import PALETTES
from .pipeline import DEFAULT_VARIANT, VARIANTS, generate, get_config, load_pipeline

MAX_SIZE = int(os.environ.get("PIXELGEN_MAX_SIZE", "256"))
SERVER_VARIANT = os.environ.get("PIXELGEN_VARIANT", DEFAULT_VARIANT)

_executor = ThreadPoolExecutor(max_workers=1)  # one inference at a time


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the pipeline once at startup so the first request isn't a cold start."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, partial(load_pipeline, SERVER_VARIANT))
    yield


app = FastAPI(title="SpriteStack PixelGen", version=__version__, lifespan=lifespan)


class SpriteRequest(BaseModel):
    prompt: Optional[str] = Field("sprite", max_length=300)
    width: Optional[int] = Field(32, ge=1, le=MAX_SIZE)
    height: Optional[int] = Field(32, ge=1, le=MAX_SIZE)
    seed: Optional[int] = Field(None, ge=0, lt=2**32)
    palette: Optional[str] = None
    variant: Optional[str] = None


@app.get("/health")
async def health():
    return {"status": "ok", "model": f"spritestack-pixelgen-{SERVER_VARIANT}", "variants": sorted(VARIANTS)}


@app.post("/generate-sprite")
async def generate_sprite(req: SpriteRequest):
    prompt = (req.prompt or "sprite").strip() or "sprite"
    width, height = req.width or 32, req.height or 32
    if req.palette and req.palette.lower().replace("-", "").replace("_", "") not in PALETTES:
        raise HTTPException(422, f"Unknown palette {req.palette!r}. Available: {sorted(PALETTES)}")
    try:
        cfg = get_config(req.variant or SERVER_VARIANT, palette=req.palette)
    except ValueError as e:
        raise HTTPException(422, str(e))

    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        _executor, partial(generate, prompt, width, height, req.seed, cfg)
    )
    return JSONResponse({
        "hex": image_to_hex(res.image),
        "width": width,
        "height": height,
        "prompt": prompt,
        "model": f"spritestack-pixelgen-{res.variant}",
        "seed": res.seed,
        "variant": res.variant,
        "palette": req.palette,
        "timings": {k: round(v, 3) for k, v in res.timings.items()},
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
