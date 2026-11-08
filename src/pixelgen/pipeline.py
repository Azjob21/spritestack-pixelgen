"""
Config-driven diffusion pipeline.

Experiments are *configs*, not copies of this file. ``VARIANTS`` is a
cumulative ladder where each row changes exactly one thing from the previous:

    fm16  native 16x16 text-conditioned flow-matching model (sprite_model.py), no SD, no VAE

    v0  baseline (legacy prompt, legacy post, seed 42 when unset)
    e1  + trigger word / clean prompt
    e2  + background keying, grid snapping, palette discipline (new post)
    e3  + CFG off (guidance 1.0)
    e4  + tiny VAE (TAESD)
    e5  + 4 steps
    e6  + 384px generation

Recommended config (measured on an i7-1065G7, CPU, 64x64 output, seed 42):

    s6  v0 prompt + grid post, 6 steps, guidance 2.5, TAESD   ~74s PyTorch, ~61s OpenVINO fp32,
        ~52s OpenVINO int8 (v0 baseline: ~96s). 4 steps loses coherence; 8 steps is the
        quality ceiling at ~93s.

Any field can be overridden ad hoc, e.g. ``generate(p, config="e2", steps=6)``,
which is how you run single-variable ablations off the ladder.

Importing this module does not import torch.
"""
from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Optional, Union

from PIL import Image

DEFAULT_MODEL_DIR = Path(os.environ.get("PIXELGEN_MODEL_DIR", "models/pixel-art-model"))
DEFAULT_LORA_DIR = Path(os.environ.get("PIXELGEN_LORA_DIR", "models/loras"))
DEFAULT_LORA_FILE = "PixelArtRedmond15V-PixelArt-PIXARFK.safetensors"
DEFAULT_FM_CKPT = Path(os.environ.get("PIXELGEN_FM_CKPT", "models/pixelgen_ema.pt"))
DEFAULT_TAESD_DIR = Path(os.environ.get("PIXELGEN_TAESD_DIR", "models/taesd"))
TAESD_HUB_ID = "madebyollin/taesd"  # SD 1.x TAESD

BACKGROUNDS = ("magenta", "green")


@dataclass(frozen=True)
class Config:
    name: str = "custom"
    # model-affecting (changing these reloads the pipeline)
    lora_scale: float = 0.9
    tiny_vae: bool = False
    # sampling
    steps: int = 8
    guidance: float = 1.8
    gen_size: int = 512
    # prompt
    prompt_style: str = "clean"      # "v0" | "v0t" (v0 + PixArFK) | "clean"
    bg: str = "magenta"              # backdrop requested from the model ("clean"+"grid" only)
    # post-processing
    post: str = "grid"               # "legacy" | "grid"
    palette: Optional[str] = None    # name from palettes.PALETTES; None = generic quantise
    num_colors: int = 32
    outline: bool = False
    bg_tol: float = 60.0             # RGB distance for backdrop keying ("grid" post only)
    edge_thr: float = 2.0            # edge-bounded flood threshold; 0 = colour keying only
    # backend: "sd" = Stable Diffusion + LoRA (above), "fm" = native 16x16 flow-matching model.
    # With "fm", steps/guidance mean flow-matching Euler steps / CFG scale; num_colors and palette
    # are applied to the 16x16 sprite; output is nearest-neighbour scaled to (width, height).
    backend: str = "sd"
    # misc
    fixed_seed: Optional[int] = None  # used only when the caller passes no seed (v0 compat)
    threads: Optional[int] = None     # CPU threads; None = PIXELGEN_THREADS or cores // 2

    def __post_init__(self) -> None:
        if self.prompt_style not in ("v0", "v0t", "clean"):
            raise ValueError(
                f"prompt_style must be 'v0', 'v0t' or 'clean', got {self.prompt_style!r}")
        if self.backend not in ("sd", "fm"):
            raise ValueError(f"backend must be 'sd' or 'fm', got {self.backend!r}")
        if self.post not in ("legacy", "grid"):
            raise ValueError(f"post must be 'legacy' or 'grid', got {self.post!r}")
        if self.bg not in BACKGROUNDS:
            raise ValueError(f"bg must be one of {BACKGROUNDS}, got {self.bg!r}")
        if self.steps < 1 or self.gen_size % 8 or self.gen_size < 128:
            raise ValueError("steps >= 1 and gen_size a multiple of 8 (>= 128) required")


_V0 = Config(name="v0", prompt_style="v0", post="legacy", fixed_seed=42)
_E1 = replace(_V0, name="e1", prompt_style="clean", fixed_seed=None)
_E2 = replace(_E1, name="e2", post="grid")
_E3 = replace(_E2, name="e3", guidance=1.0)
_E4 = replace(_E3, name="e4", tiny_vae=True)
_E5 = replace(_E4, name="e5", steps=4)
_E6 = replace(_E5, name="e6", gen_size=384)

_V0G = replace(_V0, name="v0g", post="grid", bg_tol=25.0)
_V0GT = replace(_V0G, name="v0gt", prompt_style="v0t")
_G3 = replace(_V0G, name="g3", guidance=1.0)
_G4 = replace(_G3, name="g4", tiny_vae=True)
_G5 = replace(_G4, name="g5", steps=4)
_G6 = replace(_G5, name="g6", gen_size=384)

# Rescue experiments for "CFG off collapses the style" (g3 raw outputs are washed out):
_G3T = replace(_G3, name="g3t", prompt_style="v0t")   # CFG off + trigger word
_G3L = replace(_G3, name="g3l", lora_scale=1.3)       # CFG off + stronger style LoRA
_S4 = replace(_V0G, name="s4", steps=4)               # keep CFG, halve steps instead
# Recommended: best quality per second found in the step sweep (4 -> 8 steps, guidance 2.5).
_S6 = replace(_V0G, name="s6", steps=6, guidance=2.5, tiny_vae=True)

# Native 16x16 flow-matching model trained on the Pixel Art Atlas (see sprite_model.py).
_FM16 = Config(name="fm16", backend="fm", steps=10, guidance=2.0, num_colors=32)

VARIANTS: dict[str, Config] = {
    c.name: c for c in (_V0, _E1, _E2, _E3, _E4, _E5, _E6, _V0G, _V0GT, _G3, _G4, _G5, _G6, _G3T, _G3L, _S4, _S6, _FM16)
}
DEFAULT_VARIANT = "s6"


def get_config(config: Union[None, str, Config] = None, **overrides: Any) -> Config:
    """Resolve a variant name / Config plus field overrides into a Config."""
    if config is None:
        config = DEFAULT_VARIANT
    if isinstance(config, str):
        if config not in VARIANTS:
            raise ValueError(f"Unknown variant {config!r}. Available: {sorted(VARIANTS)}")
        config = VARIANTS[config]
    overrides = {k: v for k, v in overrides.items() if v is not None}
    valid = {f.name for f in fields(Config)}
    bad = set(overrides) - valid
    if bad:
        raise ValueError(f"Unknown config fields: {sorted(bad)}")
    if overrides:
        suffix = "_".join(f"{k}{v}" for k, v in sorted(overrides.items()))
        config = replace(config, name=f"{config.name}+{suffix}", **overrides)
    return config


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
_V0_PREFIX = (
    "pixel art, 16-bit sprite, pixelated, sharp pixels, "
    "no anti-aliasing, retro game asset, flat colors, "
)
_V0_NEGATIVE = (
    "blurry, painting, photorealistic, smooth, soft edges, "
    "watercolor, oil painting, 3d render, antialiased, "
    "high resolution, detailed textures, smooth gradients, shadow shading"
)
_CLEAN_PREFIX = "PixArFK, pixel art sprite, pixelated, sharp pixels, retro game asset, flat colors, "
_CLEAN_NEGATIVE = (
    "blurry, painting, photorealistic, smooth, soft edges, watercolor, oil painting, "
    "3d render, antialiased, high resolution, detailed textures, gradient background, "
    "smooth gradients, shadow shading, multiple characters"
)


def build_prompts(cfg: Config, prompt: str) -> tuple[str, str]:
    """Return (positive, negative). With CFG off the negative is ignored by diffusers,
    so everything that matters is already phrased positively."""
    if cfg.prompt_style in ("v0", "v0t"):
        pos = (_V0_PREFIX + f"8-bit pixel-art of {prompt}, retro game sprite, "
               "solid background, NES style")
        if cfg.prompt_style == "v0t":
            pos = "PixArFK, " + pos
        return pos, _V0_NEGATIVE
    backdrop = (f"plain flat {cfg.bg} background, no shadow" if cfg.post == "grid"
                else "solid background")
    pos = (_CLEAN_PREFIX + f"single character, centered, front view, {prompt}, "
           f"{backdrop}, retro game sprite")
    return pos, _CLEAN_NEGATIVE


# --------------------------------------------------------------------------- #
# Model loading (one cached pipeline; reloads only if model-affecting fields change)
# --------------------------------------------------------------------------- #
_cache: dict[str, Any] = {"key": None, "pipe": None}


def _threads(cfg: Config) -> int:
    if cfg.threads:
        return cfg.threads
    env = os.environ.get("PIXELGEN_THREADS")
    if env:
        return int(env)
    return max(1, (os.cpu_count() or 2) // 2)  # logical -> physical cores (SMT)


_fm_cache: dict[str, Any] = {"key": None, "model": None}


def _load_fm(cfg: Config, ckpt: Path = DEFAULT_FM_CKPT):
    key = (str(ckpt), _threads(cfg))
    if _fm_cache["model"] is None or _fm_cache["key"] != key:
        from .sprite_model import SpriteModel
        if not Path(ckpt).is_file():
            raise FileNotFoundError(f"Flow-matching checkpoint not found: {ckpt} (set PIXELGEN_FM_CKPT)")
        _fm_cache.update(key=None, model=None)
        _fm_cache.update(key=key, model=SpriteModel(ckpt, threads=_threads(cfg)))
    return _fm_cache["model"]


def load_pipeline(
    config: Union[None, str, Config] = None,
    model_dir: Path = DEFAULT_MODEL_DIR,
    lora_dir: Path = DEFAULT_LORA_DIR,
    lora_file: str = DEFAULT_LORA_FILE,
):
    cfg = get_config(config)
    if cfg.backend == "fm":
        return _load_fm(cfg)
    key = (cfg.lora_scale, cfg.tiny_vae, str(model_dir), str(lora_dir), lora_file)
    if _cache["pipe"] is not None and _cache["key"] == key:
        return _cache["pipe"]

    import torch
    from diffusers import LCMScheduler, StableDiffusionPipeline

    if not Path(model_dir).is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    on_cuda = torch.cuda.is_available()
    device = "cuda" if on_cuda else "cpu"
    dtype = torch.float16 if on_cuda else torch.float32
    if not on_cuda:
        torch.set_num_threads(_threads(cfg))

    _cache.update(key=None, pipe=None)  # free the old one before loading the next
    pipe = StableDiffusionPipeline.from_pretrained(
        str(model_dir), torch_dtype=dtype, local_files_only=True,
        feature_extractor=None, safety_checker=None,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    lora_path = Path(lora_dir) / lora_file
    if lora_path.exists():
        pipe.load_lora_weights(str(lora_dir), weight_name=lora_file)
        pipe.fuse_lora(lora_scale=cfg.lora_scale)
    else:
        warnings.warn(f"LoRA file not found: {lora_path}. Running without LoRA.")

    if cfg.tiny_vae:
        from diffusers import AutoencoderTiny
        src = str(DEFAULT_TAESD_DIR) if DEFAULT_TAESD_DIR.is_dir() else TAESD_HUB_ID
        pipe.vae = AutoencoderTiny.from_pretrained(src, torch_dtype=dtype)

    pipe = pipe.to(device)
    if not on_cuda:
        pipe.unet.to(memory_format=torch.channels_last)
    pipe.set_progress_bar_config(disable=True)

    _cache.update(key=key, pipe=pipe)
    return pipe


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    image: Image.Image          # RGBA, exactly (width, height)
    seed: int
    variant: str
    timings: dict[str, float]   # unet_s, decode_s, post_s, total_s
    raw: Optional[Image.Image] = None   # decoded model output before post-processing


def _postprocess(cfg: Config, img: Image.Image, width: int, height: int) -> Image.Image:
    from . import postprocess as pp
    if cfg.post == "legacy":
        return pp.legacy_postprocess(img, width, height, cfg.num_colors)
    return pp.postprocess(img, width, height, palette=cfg.palette,
                          num_colors=cfg.num_colors, outline=cfg.outline,
                          bg_tol=cfg.bg_tol, edge_thr=cfg.edge_thr)


def _fm_finish(rgba, cfg: Config, width: int, height: int) -> Image.Image:
    """16x16 model output -> palette discipline -> RGBA PIL image of exactly (width, height)."""
    import numpy as np
    from . import postprocess as pp
    rgb, alpha = rgba[..., :3], rgba[..., 3] > 0
    pal = pp.resolve_palette(cfg.palette)
    if pal:
        rgb = pp.snap_to_palette(rgb, pal)
    elif cfg.num_colors:
        rgb = pp.quantise_opaque(rgb, alpha, cfg.num_colors)
    rgb = np.where(alpha[..., None], rgb, 0).astype(np.uint8)
    im = Image.fromarray(np.dstack([rgb, np.where(alpha, 255, 0).astype(np.uint8)]), "RGBA")
    if im.size != (width, height):
        im = im.resize((width, height), Image.Resampling.NEAREST)   # blocky upscale: no new detail
    return im


def _generate_fm(prompt: str, width: int, height: int, seed: int, cfg: Config) -> Result:
    model = _load_fm(cfg)
    rgba, tm = model.generate(prompt, seed=seed, steps=cfg.steps, guidance=cfg.guidance)
    t0 = time.perf_counter()
    image = _fm_finish(rgba, cfg, width, height)
    post = time.perf_counter() - t0
    return Result(image=image, seed=seed, variant=cfg.name, timings={
        "unet_s": tm["unet_s"], "decode_s": 0.0, "post_s": post,
        "total_s": tm["clip_s"] + tm["unet_s"] + post, "clip_s": tm["clip_s"],
    }, raw=None)


def generate(
    prompt: str,
    width: int = 32,
    height: int = 32,
    seed: Optional[int] = None,
    config: Union[None, str, Config] = None,
    **overrides: Any,
) -> Result:
    """Generate one sprite. ``overrides`` are Config fields (steps=, guidance=, palette=...)."""
    import torch

    cfg = get_config(config, **overrides)
    if seed is None:
        seed = cfg.fixed_seed if cfg.fixed_seed is not None else int.from_bytes(os.urandom(4), "little")

    if cfg.backend == "fm":
        return _generate_fm(prompt, width, height, seed, cfg)

    pipe = load_pipeline(cfg)
    pos, neg = build_prompts(cfg, prompt)
    generator = torch.Generator("cpu").manual_seed(seed)

    t0 = time.perf_counter()
    with torch.inference_mode():
        latents = pipe(
            pos, negative_prompt=neg, width=cfg.gen_size, height=cfg.gen_size,
            num_inference_steps=cfg.steps, guidance_scale=cfg.guidance,
            generator=generator, output_type="latent",
        ).images
        t1 = time.perf_counter()
        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        pil = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    t2 = time.perf_counter()

    image = _postprocess(cfg, pil, width, height)
    t3 = time.perf_counter()

    return Result(image=image, seed=seed, variant=cfg.name, timings={
        "unet_s": t1 - t0, "decode_s": t2 - t1, "post_s": t3 - t2, "total_s": t3 - t0,
    }, raw=pil)
