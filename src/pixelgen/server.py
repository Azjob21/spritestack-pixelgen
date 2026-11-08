"""
Studio-compatible FastAPI server with built-in interactive web UI.

    PIXELGEN_VARIANT=fm16 python -m pixelgen.server --port 8000

Routes:
  GET  /                 Interactive web interface for text-to-sprite testing
  GET  /health           Health check, active variant, available variants
  POST /generate-sprite  {prompt, width, height, seed?, palette?, variant?, steps?, guidance?}
                         -> {hex, data_url, width, height, prompt, model, seed, variant, palette, timings}
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from typing import Optional

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
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
    if not os.environ.get("PIXELGEN_SKIP_PRELOAD"):
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(_executor, lambda: load_pipeline(SERVER_VARIANT))
        except Exception as e:
            print(f"[warning] Preloading variant {SERVER_VARIANT!r} failed: {e}")
    yield


app = FastAPI(title="SpriteStack PixelGen", version=__version__, lifespan=lifespan)


class SpriteRequest(BaseModel):
    prompt: Optional[str] = Field("sprite", max_length=300)
    width: Optional[int] = Field(32, ge=1, le=MAX_SIZE)
    height: Optional[int] = Field(32, ge=1, le=MAX_SIZE)
    seed: Optional[int] = Field(None, ge=0, lt=2**32)
    palette: Optional[str] = None
    variant: Optional[str] = None
    steps: Optional[int] = Field(None, ge=1, le=100)
    guidance: Optional[float] = Field(None, ge=0.0, le=20.0)


INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SpriteStack PixelGen &mdash; Sprite Playground</title>
  <style>
    :root {
      --bg: #0f1117;
      --card-bg: #181b24;
      --card-border: #262a36;
      --input-bg: #11131a;
      --text: #e2e8f0;
      --text-dim: #94a3b8;
      --accent: #10b981;
      --accent-hover: #059669;
      --accent-glow: rgba(16, 185, 129, 0.2);
      --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 1rem 2rem;
      background: var(--card-bg);
      border-bottom: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    .brand h1 { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.02em; }
    .brand .tag {
      font-size: 0.75rem;
      background: var(--input-bg);
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      color: var(--accent);
      border: 1px solid var(--card-border);
      font-family: var(--font-mono);
    }
    .status-badge {
      font-size: 0.8rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
      color: var(--accent);
      font-family: var(--font-mono);
    }
    .status-badge::before {
      content: "";
      width: 8px;
      height: 8px;
      background: var(--accent);
      border-radius: 50%;
      display: inline-block;
      box-shadow: 0 0 8px var(--accent);
    }
    main {
      flex: 1;
      max-width: 1200px;
      width: 100%;
      margin: 0 auto;
      padding: 2rem;
      display: grid;
      grid-template-columns: 420px 1fr;
      gap: 2rem;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
    }
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 1.5rem;
      display: flex;
      flex-direction: column;
      gap: 1.25rem;
    }
    .field {
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .field label {
      font-size: 0.85rem;
      color: var(--text-dim);
      font-weight: 500;
    }
    input[type="text"], input[type="number"], select {
      background: var(--input-bg);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 0.65rem 0.8rem;
      border-radius: 6px;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.15s;
    }
    input[type="text"]:focus, input[type="number"]:focus, select:focus {
      border-color: var(--accent);
    }
    .presets {
      display: flex;
      flex-wrap: wrap;
      gap: 0.4rem;
      margin-top: 0.2rem;
    }
    .preset-btn {
      background: var(--input-bg);
      border: 1px solid var(--card-border);
      color: var(--text-dim);
      font-size: 0.75rem;
      padding: 0.25rem 0.55rem;
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s;
    }
    .preset-btn:hover {
      color: var(--text);
      border-color: var(--text-dim);
    }
    .row-2 {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }
    .btn-primary {
      background: var(--accent);
      color: #0b0f19;
      border: none;
      font-weight: 600;
      padding: 0.85rem 1.25rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 1rem;
      transition: background 0.15s, box-shadow 0.15s;
      display: flex;
      justify-content: center;
      align-items: center;
      gap: 0.5rem;
    }
    .btn-primary:hover:not(:disabled) {
      background: var(--accent-hover);
      box-shadow: 0 0 15px var(--accent-glow);
    }
    .btn-primary:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .preview-container {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 380px;
      background: var(--input-bg);
      border: 1px dashed var(--card-border);
      border-radius: 6px;
      position: relative;
      overflow: hidden;
    }
    .checkerboard {
      background-image: linear-gradient(45deg, #181a20 25%, transparent 25%),
                        linear-gradient(-45deg, #181a20 25%, transparent 25%),
                        linear-gradient(45deg, transparent 75%, #181a20 75%),
                        linear-gradient(-45deg, transparent 75%, #181a20 75%);
      background-size: 20px 20px;
      background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
      background-color: #121318;
    }
    .sprite-img {
      image-rendering: pixelated;
      image-rendering: -moz-crisp-edges;
      image-rendering: crisp-edges;
      transition: transform 0.1s ease;
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
    }
    .view-toolbar {
      display: flex;
      justify-content: space-between;
      width: 100%;
      align-items: center;
      margin-top: 0.5rem;
    }
    .zoom-controls {
      display: flex;
      gap: 0.3rem;
    }
    .zoom-btn {
      background: var(--input-bg);
      border: 1px solid var(--card-border);
      color: var(--text-dim);
      padding: 0.3rem 0.6rem;
      border-radius: 4px;
      font-size: 0.8rem;
      cursor: pointer;
      font-family: var(--font-mono);
    }
    .zoom-btn.active {
      color: var(--accent);
      border-color: var(--accent);
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
      gap: 0.6rem;
      width: 100%;
    }
    .metric-card {
      background: var(--input-bg);
      border: 1px solid var(--card-border);
      padding: 0.6rem;
      border-radius: 4px;
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
    }
    .metric-label { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1rem; font-family: var(--font-mono); font-weight: 600; color: var(--accent); }
    .history-section {
      margin-top: 1.5rem;
    }
    .history-title {
      font-size: 0.85rem;
      color: var(--text-dim);
      margin-bottom: 0.6rem;
    }
    .history-list {
      display: flex;
      gap: 0.6rem;
      overflow-x: auto;
      padding-bottom: 0.5rem;
    }
    .history-item {
      background: var(--input-bg);
      border: 1px solid var(--card-border);
      border-radius: 4px;
      padding: 0.4rem;
      cursor: pointer;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 0.3rem;
      flex-shrink: 0;
      transition: border-color 0.15s;
    }
    .history-item:hover { border-color: var(--accent); }
    .history-item img {
      width: 48px;
      height: 48px;
      image-rendering: pixelated;
    }
    .history-item span {
      font-size: 0.65rem;
      color: var(--text-dim);
      max-width: 60px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .spinner {
      border: 3px solid rgba(0, 0, 0, 0.1);
      width: 18px;
      height: 18px;
      border-radius: 50%;
      border-left-color: #0b0f19;
      animation: spin 0.8s linear infinite;
      display: none;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    .error-msg {
      color: #ef4444;
      background: rgba(239, 68, 68, 0.1);
      border: 1px solid rgba(239, 68, 68, 0.2);
      padding: 0.75rem;
      border-radius: 6px;
      font-size: 0.85rem;
      display: none;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <h1>SpriteStack PixelGen</h1>
      <span class="tag">Playground</span>
    </div>
    <div class="status-badge" id="backendStatus">Ready</div>
  </header>

  <main>
    <div class="panel">
      <div class="field">
        <label for="prompt">Text Prompt</label>
        <input type="text" id="prompt" value="small green slime" placeholder="e.g. red dragon, wizard with a staff...">
        <div class="presets">
          <button class="preset-btn" onclick="setPrompt('red dragon')">Red Dragon</button>
          <button class="preset-btn" onclick="setPrompt('wizard with a staff')">Wizard</button>
          <button class="preset-btn" onclick="setPrompt('small green slime')">Slime</button>
          <button class="preset-btn" onclick="setPrompt('golden treasure chest')">Treasure</button>
          <button class="preset-btn" onclick="setPrompt('purple bat')">Purple Bat</button>
        </div>
      </div>

      <div class="row-2">
        <div class="field">
          <label for="variant">Model Variant</label>
          <select id="variant">
            <option value="fm16" selected>fm16 (Native 16x16 FM, ~0.8s)</option>
            <option value="s6">s6 (SD 1.5 + LoRA, ~70s)</option>
            <option value="s4">s4 (SD 1.5 4-step)</option>
            <option value="v0">v0 (Baseline)</option>
          </select>
        </div>
        <div class="field">
          <label for="size">Output Size</label>
          <select id="size">
            <option value="16x16">16x16 (Native)</option>
            <option value="32x32" selected>32x32</option>
            <option value="64x64">64x64</option>
            <option value="128x128">128x128</option>
          </select>
        </div>
      </div>

      <div class="row-2">
        <div class="field">
          <label for="palette">Palette Constraint</label>
          <select id="palette">
            <option value="" selected>None (Full RGBA)</option>
            <option value="db32">DB32 (DawnBringer 32)</option>
            <option value="pico8">PICO-8 (16 colors)</option>
            <option value="endesga32">Endesga 32</option>
          </select>
        </div>
        <div class="field">
          <label for="seed">Seed (optional)</label>
          <div style="display: flex; gap: 0.4rem;">
            <input type="number" id="seed" placeholder="Random" style="flex: 1;">
            <button class="preset-btn" onclick="randomizeSeed()" title="Randomize">🎲</button>
          </div>
        </div>
      </div>

      <div class="row-2">
        <div class="field">
          <label for="steps">Steps</label>
          <input type="number" id="steps" value="10" min="1" max="50">
        </div>
        <div class="field">
          <label for="guidance">Guidance (CFG)</label>
          <input type="number" id="guidance" value="2.0" step="0.5" min="0" max="15">
        </div>
      </div>

      <button class="btn-primary" id="genBtn" onclick="generateSprite()">
        <span class="spinner" id="btnSpinner"></span>
        <span id="btnText">Generate Sprite</span>
      </button>

      <div class="error-msg" id="errorBox"></div>
    </div>

    <div class="panel">
      <div class="preview-container checkerboard" id="previewArea">
        <div id="placeholder" style="color: var(--text-dim); font-size: 0.9rem;">
          Enter a prompt and click <strong>Generate</strong>
        </div>
        <img id="spriteImg" class="sprite-img" style="display: none;" alt="Generated Sprite">
      </div>

      <div class="view-toolbar">
        <div class="zoom-controls">
          <button class="zoom-btn" onclick="setZoom(1)">1x</button>
          <button class="zoom-btn" onclick="setZoom(2)">2x</button>
          <button class="zoom-btn active" onclick="setZoom(4)">4x</button>
          <button class="zoom-btn" onclick="setZoom(8)">8x</button>
          <button class="zoom-btn" onclick="setZoom(12)">12x</button>
        </div>
        <button class="preset-btn" id="dlBtn" style="display:none;" onclick="downloadSprite()">Download PNG</button>
      </div>

      <div class="metrics" id="metricsArea" style="display: none;">
        <div class="metric-card">
          <span class="metric-label">Total Time</span>
          <span class="metric-value" id="mTotal">&mdash;</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">UNet / Denoise</span>
          <span class="metric-value" id="mUnet">&mdash;</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">CLIP Encode</span>
          <span class="metric-value" id="mClip">&mdash;</span>
        </div>
        <div class="metric-card">
          <span class="metric-label">Seed</span>
          <span class="metric-value" id="mSeed">&mdash;</span>
        </div>
      </div>

      <div class="history-section" id="historySection" style="display: none;">
        <div class="history-title">Recent Generations</div>
        <div class="history-list" id="historyList"></div>
      </div>
    </div>
  </main>

  <script>
    let currentZoom = 4;
    let currentDataUrl = null;
    let currentPrompt = "";

    function setPrompt(p) {
      document.getElementById('prompt').value = p;
    }

    function randomizeSeed() {
      document.getElementById('seed').value = Math.floor(Math.random() * 1000000);
    }

    function setZoom(z) {
      currentZoom = z;
      document.querySelectorAll('.zoom-btn').forEach(b => b.classList.remove('active'));
      event.target.classList.add('active');
      const img = document.getElementById('spriteImg');
      if (img.src) {
        img.style.transform = `scale(${z})`;
      }
    }

    function downloadSprite() {
      if (!currentDataUrl) return;
      const a = document.createElement('a');
      a.href = currentDataUrl;
      const clean = currentPrompt.replace(/[^a-zA-Z0-9_-]/g, '_').slice(0, 30);
      a.download = `sprite_${clean || 'out'}.png`;
      a.click();
    }

    // Auto-update steps/guidance defaults when switching variants
    document.getElementById('variant').addEventListener('change', (e) => {
      const v = e.target.value;
      if (v === 'fm16') {
        document.getElementById('steps').value = 10;
        document.getElementById('guidance').value = 2.0;
      } else if (v === 's6') {
        document.getElementById('steps').value = 6;
        document.getElementById('guidance').value = 2.5;
      }
    });

    document.getElementById('prompt').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') generateSprite();
    });

    async function generateSprite() {
      const prompt = document.getElementById('prompt').value.trim();
      const variant = document.getElementById('variant').value;
      const [w, h] = document.getElementById('size').value.split('x').map(Number);
      const palette = document.getElementById('palette').value || null;
      const seedVal = document.getElementById('seed').value;
      const seed = seedVal ? parseInt(seedVal, 10) : null;
      const steps = parseInt(document.getElementById('steps').value, 10) || null;
      const guidance = parseFloat(document.getElementById('guidance').value) || null;

      const btn = document.getElementById('genBtn');
      const spinner = document.getElementById('btnSpinner');
      const btnText = document.getElementById('btnText');
      const errorBox = document.getElementById('errorBox');
      const backendStatus = document.getElementById('backendStatus');

      errorBox.style.display = 'none';
      btn.disabled = true;
      spinner.style.display = 'inline-block';
      btnText.textContent = 'Generating...';
      backendStatus.textContent = 'Running inference...';

      try {
        const res = await fetch('/generate-sprite', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt, variant, width: w, height: h, palette, seed, steps, guidance })
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({ detail: res.statusText }));
          throw new Error(err.detail || 'Generation failed');
        }
        const data = await res.json();
        currentDataUrl = data.data_url;
        currentPrompt = prompt;

        const img = document.getElementById('spriteImg');
        img.src = data.data_url;
        img.style.display = 'block';
        img.style.transform = `scale(${currentZoom})`;
        document.getElementById('placeholder').style.display = 'none';
        document.getElementById('dlBtn').style.display = 'inline-block';

        // Update metrics
        const tm = data.timings || {};
        document.getElementById('mTotal').textContent = `${(tm.total_s || 0).toFixed(2)}s`;
        document.getElementById('mUnet').textContent = `${(tm.unet_s || 0).toFixed(2)}s`;
        document.getElementById('mClip').textContent = `${(tm.clip_s || 0).toFixed(3)}s`;
        document.getElementById('mSeed').textContent = data.seed;
        document.getElementById('metricsArea').style.display = 'grid';

        // Add to history
        addToHistory(data.data_url, prompt, data.seed);
        backendStatus.textContent = `Generated in ${(tm.total_s || 0).toFixed(2)}s`;
      } catch (err) {
        errorBox.textContent = err.message;
        errorBox.style.display = 'block';
        backendStatus.textContent = 'Error';
      } finally {
        btn.disabled = false;
        spinner.style.display = 'none';
        btnText.textContent = 'Generate Sprite';
      }
    }

    function addToHistory(dataUrl, prompt, seed) {
      const section = document.getElementById('historySection');
      const list = document.getElementById('historyList');
      section.style.display = 'block';
      const item = document.createElement('div');
      item.className = 'history-item';
      item.innerHTML = `<img src="${dataUrl}" alt="${prompt}"><span title="${prompt}">${prompt}</span>`;
      item.onclick = () => {
        currentDataUrl = dataUrl;
        currentPrompt = prompt;
        const img = document.getElementById('spriteImg');
        img.src = dataUrl;
        img.style.display = 'block';
        img.style.transform = `scale(${currentZoom})`;
        document.getElementById('placeholder').style.display = 'none';
        document.getElementById('dlBtn').style.display = 'inline-block';
      };
      list.prepend(item);
    }

    // Populate variants dynamically from /health
    fetch('/health').then(r => r.json()).then(d => {
      if (d.variants && d.variants.length) {
        const sel = document.getElementById('variant');
        const curr = sel.value;
        sel.innerHTML = '';
        d.variants.forEach(v => {
          const opt = document.createElement('option');
          opt.value = v;
          opt.textContent = v === 'fm16' ? 'fm16 (Native 16x16 FM, ~0.8s)' :
                            v === 's6' ? 's6 (SD 1.5 + LoRA, ~70s)' : v;
          if (v === 'fm16' || (curr === v && !d.variants.includes('fm16'))) opt.selected = true;
          sel.appendChild(opt);
        });
      }
    }).catch(() => {});
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    """Interactive web UI for testing text-to-sprite models."""
    return HTMLResponse(INDEX_HTML)


@app.get("/health")
async def health():
    return {"status": "ok", "model": f"spritestack-pixelgen-{SERVER_VARIANT}", "variants": sorted(VARIANTS)}


@app.post("/generate-sprite")
async def generate_sprite(req: SpriteRequest):
    prompt = (req.prompt or "sprite").strip() or "sprite"
    width, height = req.width or 32, req.height or 32
    if req.palette and req.palette.lower().replace("-", "").replace("_", "") not in PALETTES:
        raise HTTPException(422, f"Unknown palette {req.palette!r}. Available: {sorted(PALETTES)}")

    overrides = {}
    if req.palette:
        overrides["palette"] = req.palette
    if req.steps is not None:
        overrides["steps"] = req.steps
    if req.guidance is not None:
        overrides["guidance"] = req.guidance

    try:
        cfg = get_config(req.variant or SERVER_VARIANT, **overrides)
    except ValueError as e:
        raise HTTPException(422, str(e))

    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        _executor, partial(generate, prompt, width, height, req.seed, cfg)
    )

    buf = io.BytesIO()
    res.image.save(buf, format="PNG")
    data_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

    return JSONResponse({
        "hex": image_to_hex(res.image),
        "data_url": data_url,
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
