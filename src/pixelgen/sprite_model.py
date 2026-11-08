"""
Native 16x16 text-to-sprite model (flow matching + CLIP text conditioning).

    from pixelgen.sprite_model import SpriteModel
    m = SpriteModel("models/pixelgen_ema.pt")
    rgba, timings = m.generate("small green slime", seed=1)      # uint8 (16, 16, 4)

The module-level class/layer definitions match the training notebook exactly, so the
checkpoint's state_dict keys line up. Imported lazily by ``pipeline.py`` (needs torch + transformers).

Offline use: run once on a machine with internet
    python -m pixelgen.sprite_model --export-clip models/clip
then set PIXELGEN_CLIP_DIR=models/clip.
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelCfg:
    size: int = 16
    base: int = 96
    mult: tuple = (1, 2, 3)
    n_blocks: int = 2
    attn_res: tuple = (8, 4)
    xattn_res: tuple = (16, 8, 4)
    heads: int = 4
    text_model: str = "openai/clip-vit-base-patch32"
    max_len: int = 32


# --------------------------------------------------------------------------- #
# Network (identical to the training notebook)
# --------------------------------------------------------------------------- #
def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = (t[:, None] * 1000) * freqs[None]
    return torch.cat([a.cos(), a.sin()], -1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, emb):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.e = nn.Linear(emb, cout * 2)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.e(F.silu(e))[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + scale) + shift
        return self.c2(F.silu(h)) + self.skip(x)


class Attn(nn.Module):
    def __init__(self, c, heads):
        super().__init__()
        self.n, self.qkv, self.o, self.h = nn.GroupNorm(8, c), nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1), heads

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.h, C // self.h, H * W).transpose(-1, -2).unbind(1)
        o = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(B, C, H, W)
        return x + self.o(o)


class CrossAttn(nn.Module):
    def __init__(self, c, ctx_dim, heads):
        super().__init__()
        self.n = nn.GroupNorm(8, c)
        self.q, self.k, self.v, self.o = nn.Linear(c, c), nn.Linear(ctx_dim, c), nn.Linear(ctx_dim, c), nn.Linear(c, c)
        self.h = heads

    def forward(self, x, ctx, mask):
        B, C, H, W = x.shape
        h = self.n(x).flatten(2).transpose(1, 2)
        q = self.q(h).view(B, H * W, self.h, C // self.h).transpose(1, 2)
        k = self.k(ctx).view(B, -1, self.h, C // self.h).transpose(1, 2)
        v = self.v(ctx).view(B, -1, self.h, C // self.h).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        o = self.o(o.transpose(1, 2).reshape(B, H * W, C))
        return x + o.transpose(1, 2).reshape(B, C, H, W)


def upsample(c):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1))


class UNet(nn.Module):
    def __init__(self, cfg: ModelCfg, ctx_dim: int = 512, in_ch: int = 4):
        super().__init__()
        ch = [cfg.base * m for m in cfg.mult]
        emb = cfg.base * 4
        self.base = cfg.base
        self.t_mlp = nn.Sequential(nn.Linear(cfg.base, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.pool = nn.Linear(ctx_dim, emb)
        self.inp = nn.Conv2d(in_ch, ch[0], 3, padding=1)

        def extras(res, c):
            L = []
            if res in cfg.attn_res: L.append(Attn(c, cfg.heads))
            if res in cfg.xattn_res: L.append(CrossAttn(c, ctx_dim, cfg.heads))
            return L

        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        res, c, skips = cfg.size, ch[0], [ch[0]]
        for i, co in enumerate(ch):
            for _ in range(cfg.n_blocks):
                blk = nn.ModuleList([ResBlock(c, co, emb)]); c = co
                blk.extend(extras(res, c))
                self.down.append(blk); skips.append(c)
            if i < len(ch) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1)); skips.append(c); res //= 2
        self.mid = nn.ModuleList([ResBlock(c, c, emb), Attn(c, cfg.heads), CrossAttn(c, ctx_dim, cfg.heads),
                                  ResBlock(c, c, emb)])
        for i, co in reversed(list(enumerate(ch))):
            for j in range(cfg.n_blocks + 1):
                blk = nn.ModuleList([ResBlock(c + skips.pop(), co, emb)]); c = co
                blk.extend(extras(res, c))
                if i > 0 and j == cfg.n_blocks:
                    blk.append(upsample(c)); res *= 2
                self.up.append(blk)
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, in_ch, 3, padding=1))

    def forward(self, x, t, cond):
        ctx, mask, pooled = cond
        ctx, pooled = ctx.to(x.dtype), pooled.to(x.dtype)
        e = self.t_mlp(timestep_embedding(t, self.base)) + self.pool(pooled)

        def run(blk, h):
            h = blk[0](h, e)
            for extra in blk[1:]:
                h = extra(h, ctx, mask) if isinstance(extra, CrossAttn) else extra(h)
            return h

        h = self.inp(x); hs = [h]
        for item in self.down:
            h = run(item, h) if isinstance(item, nn.ModuleList) else item(h)
            hs.append(h)
        h = self.mid[0](h, e); h = self.mid[1](h); h = self.mid[2](h, ctx, mask); h = self.mid[3](h, e)
        for blk in self.up:
            h = blk[0](torch.cat([h, hs.pop()], 1), e)
            for extra in blk[1:]:
                h = extra(h, ctx, mask) if isinstance(extra, CrossAttn) else extra(h)
        return self.out(h)


# --------------------------------------------------------------------------- #
# Inference wrapper
# --------------------------------------------------------------------------- #
def _clip_source(name: str) -> tuple[str, bool]:
    """Local export dir (PIXELGEN_CLIP_DIR or models/clip) if present, else the hub id."""
    for cand in (os.environ.get("PIXELGEN_CLIP_DIR"), "models/clip"):
        if cand and Path(cand).is_dir():
            return str(cand), True
    return name, False


def export_clip(dest: str, name: str = ModelCfg.text_model) -> None:
    """Download the CLIP text encoder + tokenizer once and save them for offline use."""
    from transformers import CLIPTextModel, CLIPTokenizerFast
    CLIPTokenizerFast.from_pretrained(name).save_pretrained(dest)
    CLIPTextModel.from_pretrained(name).save_pretrained(dest)
    print(f"saved CLIP text encoder to {dest}; set PIXELGEN_CLIP_DIR={dest}")


class SpriteModel:
    def __init__(self, ckpt: str | Path, device: Optional[str] = None, threads: Optional[int] = None,
                 clip_dir: Optional[str] = None):
        from transformers import CLIPTextModel, CLIPTokenizerFast
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu" and threads:
            torch.set_num_threads(threads)

        ck = torch.load(str(ckpt), map_location="cpu")
        known = {f.name for f in fields(ModelCfg)}
        self.cfg = ModelCfg(**{k: (tuple(v) if isinstance(v, list) else v)
                               for k, v in ck["cfg"].items() if k in known})
        self.step = ck.get("step")

        self.net = UNet(self.cfg).eval()
        self.net.load_state_dict({k: v.float() for k, v in ck["ema"].items()})   # stored fp16 -> fp32
        self.net.to(self.device).requires_grad_(False)

        src, local = (clip_dir, True) if clip_dir else _clip_source(self.cfg.text_model)
        try:
            # Try offline cache first for instant loading without network retries
            self.tok = CLIPTokenizerFast.from_pretrained(src, local_files_only=True)
            self.clip = CLIPTextModel.from_pretrained(src, local_files_only=True).to(self.device).eval().requires_grad_(False)
        except Exception:
            self.tok = CLIPTokenizerFast.from_pretrained(src, local_files_only=local)
            self.clip = CLIPTextModel.from_pretrained(src, local_files_only=local).to(self.device).eval().requires_grad_(False)
        with torch.inference_mode():
            self._null = self._encode("")                       # empty prompt = unconditional branch

    @torch.inference_mode()
    def _encode(self, prompt: str):
        t = self.tok(prompt, truncation=True, max_length=self.cfg.max_len, return_tensors="pt")
        ids = t["input_ids"].to(self.device)
        h = self.clip(input_ids=ids, attention_mask=t["attention_mask"].to(self.device)).last_hidden_state
        mask = torch.ones(h.shape[:2], dtype=torch.bool, device=self.device)   # no padding -> all real tokens
        return h, mask, h[:, -1]                                # pooled = EOS (last real token)

    @torch.inference_mode()
    def generate(self, prompt: str, seed: Optional[int] = None, steps: int = 20, guidance: float = 3.0):
        """Returns (uint8 RGBA (S, S, 4), timings dict with clip_s and unet_s)."""
        S = self.cfg.size
        t0 = time.perf_counter()
        ctx_c, mask_c, pool_c = self._encode(prompt)
        t1 = time.perf_counter()

        use_cfg = guidance != 1.0
        if use_cfg:                                             # pad the 2-token null prompt to the prompt's length
            ctx_u, mask_u, pool_u = self._null
            pad = ctx_c.shape[1] - ctx_u.shape[1]
            ctx_u = F.pad(ctx_u, (0, 0, 0, pad)); mask_u = F.pad(mask_u, (0, pad), value=False)
            cond = (torch.cat([ctx_c, ctx_u]), torch.cat([mask_c, mask_u]), torch.cat([pool_c, pool_u]))
        else:
            cond = (ctx_c, mask_c, pool_c)

        g = torch.Generator("cpu")
        g.manual_seed(int(seed)) if seed is not None else g.seed()
        x = torch.randn(1, 4, S, S, generator=g).to(self.device)     # CPU noise: same seed -> same sprite on any device
        ts = torch.linspace(0, 1, steps + 1, device=self.device)
        for i in range(steps):
            t = ts[i].expand(2 if use_cfg else 1)
            if use_cfg:
                vc, vu = self.net(torch.cat([x, x]), t, cond).chunk(2)
                v = vu + guidance * (vc - vu)
            else:
                v = self.net(x, t, cond)
            x = x + (ts[i + 1] - ts[i]) * v
        t2 = time.perf_counter()

        x = x.float().clamp(-1, 1).cpu()
        rgb = ((x[0, :3] + 1) * 127.5).round().permute(1, 2, 0).numpy()
        a = (x[0, 3] > 0).numpy()
        rgb = np.where(a[..., None], rgb, 0).astype(np.uint8)
        rgba = np.concatenate([rgb, (a * 255).astype(np.uint8)[..., None]], -1)
        return rgba, {"clip_s": t1 - t0, "unet_s": t2 - t1}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-clip", metavar="DIR", help="save the CLIP text encoder for offline use")
    args = ap.parse_args()
    if args.export_clip:
        export_clip(args.export_clip)
