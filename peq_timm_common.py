"""Shared PEQ model and timm-based ImageNet training utilities."""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import ImageFilter, ImageOps
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from timm.data import Mixup, create_transform
from timm.data.distributed_sampler import RepeatAugSampler
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.layers import DropPath
from timm.loss import BinaryCrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import ModelEmaV3, NativeScaler, dispatch_clip_grad
from timm.utils import init_distributed_device, is_primary, random_seed, setup_default_logging


LOG = logging.getLogger("peq_timm")
DATA_MODES = {"tied_data", "tied_data_rec"}
VALID_MODES = {"single", "tied", "untied", "tied_data", "tied_data_rec"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
LoraCache = dict[int, tuple[int, torch.Tensor]]


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    dim: int = 384
    n_reg: int = 8
    heads: int = 6
    steps: int = 4
    num_classes: int = 1000
    mode: str = "tied"
    attention: str = "softmax"
    func_k: int = 64
    func_lambda: float = 1.0
    lora_rank: int = 0
    lora_alpha: float = 1.0
    lora_compose: str = "indep"
    lora_dz: int = 64
    rmsnorm: bool = False
    layerscale: bool = False
    layerscale_init: float = 1e-4
    qk_rms_norm: bool = True
    qk_rms_eps: float = 1e-6
    rope: bool = False
    rope_base: float = 100.0
    reg_rope: bool = True
    reg_rope_theta: float = 10000.0
    gamma_d: float = 0.5
    drop_path: float = 0.05

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode={self.mode}; choose from {sorted(VALID_MODES)}")
        if self.attention not in {"softmax", "funcattn"}:
            raise ValueError("attention must be softmax or funcattn")
        if self.lora_compose not in {"indep", "cumul", "recur"}:
            raise ValueError("lora_compose must be indep, cumul, or recur")
        if self.dim % self.heads:
            raise ValueError(f"dim={self.dim} must be divisible by heads={self.heads}")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.steps < 1 or self.n_reg < 1:
            raise ValueError("steps and n_reg must be positive")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        aliases = {
            "img": "image_size",
            "patch": "patch_size",
            "D": "dim",
            "n_reg": "n_reg",
            "heads": "heads",
            "T": "steps",
            "attn": "attention",
            "funck": "func_k",
            "func_lambda": "func_lambda",
            "lora_rank": "lora_rank",
            "lora_alpha": "lora_alpha",
            "lora_compose": "lora_compose",
            "lora_dz": "lora_dz",
            "rmsnorm": "rmsnorm",
            "layerscale": "layerscale",
            "ls_init": "layerscale_init",
            "qk_rms_norm": "qk_rms_norm",
            "qk_rms_eps": "qk_rms_eps",
            "rope": "rope",
            "rope_base": "rope_base",
            "reg_rope": "reg_rope",
            "reg_rope_theta": "reg_rope_theta",
            "gamma_d0": "gamma_d",
        }
        fields = cls.__dataclass_fields__
        normalized: dict[str, Any] = {}
        for key, value in values.items():
            target = aliases.get(key, key)
            if target in fields:
                normalized[target] = value
        return cls(**normalized)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xf.to(dtype) * self.weight.to(dtype)


def _norm(cfg: ModelConfig, dim: int) -> nn.Module:
    return RMSNorm(dim, cfg.qk_rms_eps) if cfg.rmsnorm else nn.LayerNorm(dim)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _build_2d_rope(grid: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 4:
        raise ValueError(f"2D axial RoPE requires head_dim divisible by 4, got {head_dim}")
    dim = head_dim // 2
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(grid).float()
    freq = torch.outer(positions, freqs).repeat_interleave(2, dim=-1)
    fy = freq[:, None, :].expand(grid, grid, -1)
    fx = freq[None, :, :].expand(grid, grid, -1)
    phases = torch.cat([fy, fx], dim=-1).reshape(grid * grid, head_dim)
    return phases.cos()[None, None], phases.sin()[None, None]


def _build_1d_rope(n: int, head_dim: int, base: float) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2:
        raise ValueError(f"1D RoPE requires even head_dim, got {head_dim}")
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    phases = torch.outer(torch.arange(n).float(), freqs).repeat_interleave(2, dim=-1)
    return phases.cos()[None, None], phases.sin()[None, None]


def _apply_rope(
    x: torch.Tensor,
    rope: Optional[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if rope is None:
        return x
    cos, sin = rope
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)
    return x * cos + _rotate_half(x) * sin


class LoraLinear(nn.Module):
    def __init__(self, cfg: ModelConfig, fin: int, fout: int, rank: int):
        super().__init__()
        self.base = nn.Linear(fin, fout)
        self.rank = rank
        self.compose = cfg.lora_compose
        self.scale = cfg.lora_alpha / rank if rank else 1.0
        if rank > 0:
            if self.compose == "recur":
                self.A = nn.Parameter(torch.randn(rank, fin) * (1.0 / math.sqrt(fin)))
                self.B = nn.Parameter(torch.zeros(fout, rank))
                self.gate = nn.Linear(cfg.lora_dz, rank)
            else:
                self.A = nn.Parameter(torch.randn(cfg.steps - 1, rank, fin) * (1.0 / math.sqrt(fin)))
                self.B = nn.Parameter(torch.zeros(cfg.steps - 1, fout, rank))

    def forward(
        self,
        x: torch.Tensor,
        t: int = 0,
        z: Optional[torch.Tensor] = None,
        cache: Optional[LoraCache] = None,
    ) -> torch.Tensor:
        out = self.base(x)
        if self.rank == 0:
            return out
        if self.compose == "recur":
            if z is None:
                raise ValueError("lora_compose=recur requires recurrent state z")
            gate = self.gate(z)
            return out + (((x @ self.A.T) * gate[:, None, :]) @ self.B.T) * self.scale
        if t > 0:
            if self.compose == "cumul":
                key = id(self)
                cached = cache.get(key) if cache is not None else None
                last_t, delta = cached if cached is not None else (0, None)
                if t < last_t:
                    raise ValueError("cumulative LoRA cache received decreasing step indices")
                if t > last_t:
                    new_delta = torch.einsum(
                        "sor,sri->oi", self.B[last_t:t], self.A[last_t:t]
                    )
                    delta = new_delta if delta is None else delta + new_delta
                    if cache is not None:
                        cache[key] = (t, delta)
                out = out + F.linear(x, delta) * self.scale
            else:
                out = out + ((x @ self.A[t - 1].T) @ self.B[t - 1].T) * self.scale
        return out


class QKRMSMultiheadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, lora_rank: int):
        super().__init__()
        self.dim = cfg.dim
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.q_proj = LoraLinear(cfg, cfg.dim, cfg.dim, lora_rank)
        self.k_proj = LoraLinear(cfg, cfg.dim, cfg.dim, lora_rank)
        self.v_proj = LoraLinear(cfg, cfg.dim, cfg.dim, lora_rank)
        self.out_proj = LoraLinear(cfg, cfg.dim, cfg.dim, lora_rank)
        if cfg.qk_rms_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.qk_rms_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.qk_rms_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        return x.view(bsz, seq, self.heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq, self.dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        t: int = 0,
        z: Optional[torch.Tensor] = None,
        rope_q: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        rope_k: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        lora_cache: Optional[LoraCache] = None,
    ) -> torch.Tensor:
        q = self.q_norm(self._split_heads(self.q_proj(query, t, z, lora_cache)))
        k = self.k_norm(self._split_heads(self.k_proj(key, t, z, lora_cache)))
        v = self._split_heads(self.v_proj(value, t, z, lora_cache))
        q, k = _apply_rope(q, rope_q), _apply_rope(k, rope_k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        return self.out_proj(self._merge_heads(out), t, z, lora_cache)


class FuncAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, lora_rank: int):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.heads = cfg.heads
        self.head_dim = cfg.head_dim
        self.q_proj = LoraLinear(cfg, cfg.dim, cfg.dim, lora_rank)
        self.k_proj = LoraLinear(cfg, cfg.dim, cfg.dim, lora_rank)
        self.v_proj = LoraLinear(cfg, cfg.dim, cfg.dim, lora_rank)
        self.out_proj = LoraLinear(cfg, cfg.dim, cfg.dim, lora_rank)
        self.phi = LoraLinear(cfg, cfg.dim, cfg.heads * cfg.func_k, lora_rank)
        self.psi = LoraLinear(cfg, cfg.dim, cfg.heads * cfg.func_k, lora_rank)
        if cfg.qk_rms_norm:
            self.q_norm = RMSNorm(self.head_dim, cfg.qk_rms_eps)
            self.k_norm = RMSNorm(self.head_dim, cfg.qk_rms_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        return x.view(bsz, seq, self.heads, self.head_dim).transpose(1, 2)

    def _basis(
        self,
        layer: LoraLinear,
        x: torch.Tensor,
        t: int,
        z: Optional[torch.Tensor],
        lora_cache: Optional[LoraCache],
    ) -> torch.Tensor:
        bsz, seq, _ = x.shape
        basis = layer(x, t, z, lora_cache).view(
            bsz, seq, self.heads, self.cfg.func_k
        ).softmax(dim=-1)
        return basis.permute(0, 2, 1, 3)

    def _solve_transport(self, mat: torch.Tensor, rhs: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        mat = torch.nan_to_num(mat, nan=0.0, posinf=1e4, neginf=-1e4)
        rhs = torch.nan_to_num(rhs, nan=0.0, posinf=1e4, neginf=-1e4)
        solve_mat = mat.cpu() if mat.device.type == "mps" else mat
        solve_rhs = rhs.cpu() if rhs.device.type == "mps" else rhs
        sol, info = torch.linalg.solve_ex(solve_mat, solve_rhs)
        if info.any():
            eye = torch.eye(self.cfg.func_k, device=solve_mat.device, dtype=solve_mat.dtype)
            retry = solve_mat + max(self.cfg.func_lambda, 1.0) * 1e-3 * eye
            sol, info = torch.linalg.solve_ex(retry, solve_rhs)
            if info.any():
                sol = torch.linalg.pinv(retry) @ solve_rhs
        if mat.device.type == "mps":
            sol = sol.to(mat.device)
        return sol.transpose(-1, -2).to(out_dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        t: int = 0,
        z: Optional[torch.Tensor] = None,
        rope_q: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        rope_k: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        lora_cache: Optional[LoraCache] = None,
    ) -> torch.Tensor:
        bsz, n_query, _ = query.shape
        q = _apply_rope(
            self.q_norm(self._split_heads(self.q_proj(query, t, z, lora_cache))), rope_q
        )
        k = _apply_rope(
            self.k_norm(self._split_heads(self.k_proj(key, t, z, lora_cache))), rope_k
        )
        v = self._split_heads(self.v_proj(value, t, z, lora_cache))
        phi = self._basis(self.phi, query, t, z, lora_cache)
        psi = self._basis(self.psi, key, t, z, lora_cache)
        with torch.autocast(device_type=query.device.type, enabled=False):
            phi_f, psi_f = phi.float(), psi.float()
            q_t = phi_f.transpose(-1, -2) @ q.float()
            k_t = psi_f.transpose(-1, -2) @ k.float()
            v_t = psi_f.transpose(-1, -2) @ v.float()
            gram = k_t @ k_t.transpose(-1, -2)
            eye = torch.eye(self.cfg.func_k, device=gram.device, dtype=gram.dtype)
            gram = gram + self.cfg.func_lambda * eye
            qk_t = q_t @ k_t.transpose(-1, -2)
            transport = self._solve_transport(gram, qk_t.transpose(-1, -2), q.dtype)
        ctx = phi @ (transport @ v_t.to(q.dtype))
        ctx = ctx.transpose(1, 2).contiguous().view(bsz, n_query, self.dim)
        return self.out_proj(ctx, t, z, lora_cache)


def _make_attention(cfg: ModelConfig, lora_rank: int) -> nn.Module:
    if cfg.attention == "softmax":
        return QKRMSMultiheadAttention(cfg, lora_rank)
    return FuncAttention(cfg, lora_rank)


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, lora_rank: int = 0, layerscale_steps: int = 1):
        super().__init__()
        if layerscale_steps < 1:
            raise ValueError("layerscale_steps must be positive")
        self.cfg = cfg
        self.lnc = _norm(cfg, cfg.dim)
        self.compress = _make_attention(cfg, lora_rank)
        self.lnr = _norm(cfg, cfg.dim)
        self.refine = _make_attention(cfg, lora_rank)
        self.lnb = _norm(cfg, cfg.dim)
        self.broadcast = _make_attention(cfg, lora_rank)
        self.lnm = _norm(cfg, cfg.dim)
        self.w1 = LoraLinear(cfg, cfg.dim, 4 * cfg.dim, lora_rank)
        self.w2 = LoraLinear(cfg, 4 * cfg.dim, cfg.dim, lora_rank)
        self.gamma_d = nn.Parameter(torch.tensor(cfg.gamma_d))
        self.drop_path = DropPath(cfg.drop_path) if cfg.drop_path > 0 else nn.Identity()
        if cfg.layerscale:
            # Weight tying applies to the block, not to LayerScale: each
            # iteration gets an independently learnable row.
            scale_shape = (
                (cfg.dim,) if layerscale_steps == 1 else (layerscale_steps, cfg.dim)
            )
            make_scale = lambda: nn.Parameter(
                cfg.layerscale_init * torch.ones(scale_shape)
            )
            self.ls_c = make_scale()
            self.ls_rf = make_scale()
            self.ls_d = make_scale()
            self.ls_b = make_scale()
            self.ls_m = make_scale()

    def _scale(self, name: str, value: torch.Tensor, t: int) -> torch.Tensor:
        if not self.cfg.layerscale:
            return value
        scale = getattr(self, name)
        if scale.ndim == 2:
            scale = scale[t]
        return scale * value

    def _residual(self, name: str, value: torch.Tensor, t: int) -> torch.Tensor:
        return self.drop_path(self._scale(name, value, t))

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        data: bool,
        t: int = 0,
        z: Optional[torch.Tensor] = None,
        rope_p: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        rope_r: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        lora_cache: Optional[LoraCache] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lx = self.lnc(x)
        r = r + self._residual(
            "ls_c",
            self.compress(
                self.lnr(r), lx, lx, t, z, rope_q=rope_r, rope_k=rope_p,
                lora_cache=lora_cache,
            ),
            t,
        )
        r = r + self._residual(
            "ls_rf",
            self.refine(
                self.lnr(r), self.lnr(r), self.lnr(r), t, z,
                rope_q=rope_r, rope_k=rope_r, lora_cache=lora_cache,
            ),
            t,
        )
        bx = self.lnb(x)
        xhat = self.broadcast(
            bx, self.lnr(r), self.lnr(r), t, z,
            rope_q=rope_p, rope_k=rope_r, lora_cache=lora_cache,
        )
        eps = bx - xhat
        if data:
            correction = self.gamma_d * self.compress(
                self.lnr(r), eps, eps, t, z, rope_q=rope_r, rope_k=rope_p,
                lora_cache=lora_cache,
            )
            r = r + self._residual("ls_d", correction, t)
        x = x + self._residual("ls_b", xhat, t)
        mlp = self.w2(
            F.gelu(self.w1(self.lnm(x), t, z, lora_cache)), t, z, lora_cache
        )
        x = x + self._residual("ls_m", mlp, t)
        return x, r, eps.pow(2).mean()


class Net(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mode = cfg.mode
        self.data = cfg.mode in DATA_MODES
        self.T = 1 if cfg.mode == "single" else cfg.steps
        self.use_lora = cfg.lora_rank > 0 and cfg.mode not in {"single", "untied"}
        self.recur_lora = self.use_lora and cfg.lora_compose == "recur"
        self.cumul_lora = self.use_lora and cfg.lora_compose == "cumul"
        self.patch = nn.Conv2d(3, cfg.dim, cfg.patch_size, cfg.patch_size)
        self.pos = nn.Parameter(torch.randn(1, cfg.num_patches, cfg.dim) * 0.02)
        self.r0 = nn.Parameter(torch.randn(1, cfg.n_reg, cfg.dim) * 0.02)
        if cfg.rope:
            grid = cfg.image_size // cfg.patch_size
            rope_cos, rope_sin = _build_2d_rope(grid, cfg.head_dim, cfg.rope_base)
            self.register_buffer("rope_cos", rope_cos)
            self.register_buffer("rope_sin", rope_sin)
        if cfg.reg_rope:
            rrope_cos, rrope_sin = _build_1d_rope(cfg.n_reg, cfg.head_dim, cfg.reg_rope_theta)
            self.register_buffer("rrope_cos", rrope_cos)
            self.register_buffer("rrope_sin", rrope_sin)
        if cfg.mode == "untied":
            self.blocks = nn.ModuleList([Block(cfg, 0) for _ in range(self.T)])
        else:
            self.block = Block(
                cfg,
                cfg.lora_rank if self.use_lora else 0,
                layerscale_steps=self.T,
            )
        if self.recur_lora:
            self.z0 = nn.Parameter(torch.randn(cfg.lora_dz) * 0.02)
            self.projR = nn.Linear(cfg.dim, cfg.lora_dz)
            self.gru = nn.GRUCell(cfg.lora_dz, cfg.lora_dz)
        self.head = nn.Sequential(_norm(cfg, cfg.dim), nn.Linear(cfg.dim, cfg.num_classes))

    def _rope(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        return (self.rope_cos, self.rope_sin) if self.cfg.rope else None

    def _reg_rope(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        return (self.rrope_cos, self.rrope_sin) if self.cfg.reg_rope else None

    def forward(
        self,
        im: torch.Tensor,
        log: bool = False,
    ) -> tuple[torch.Tensor, list[float], list[torch.Tensor]]:
        x = self.patch(im).flatten(2).transpose(1, 2) + self.pos
        r = self.r0.expand(im.size(0), -1, -1).contiguous()
        z = self.z0.expand(im.size(0), -1).contiguous() if self.recur_lora else None
        lora_cache: Optional[LoraCache] = {} if self.cumul_lora else None
        rope_p, rope_r = self._rope(), self._reg_rope()
        resid: list[float] = []
        recon: list[torch.Tensor] = []
        for t in range(self.T):
            block = self.blocks[t] if self.mode == "untied" else self.block
            r_prev = r
            x, r, eps2 = block(x, r, self.data, t, z, rope_p, rope_r, lora_cache)
            recon.append(eps2)
            if log:
                numerator = (r - r_prev).norm(dim=-1).mean().item()
                denominator = r.norm(dim=-1).mean().item() + 1e-8
                resid.append(numerator / denominator)
            if self.recur_lora:
                z = self.gru(self.projR(r.mean(1)), z)
        return self.head(r.mean(1)), resid, recon


class NumericImageFolder(datasets.ImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        classes = [entry.name for entry in os.scandir(directory) if entry.is_dir()]
        if not classes:
            raise FileNotFoundError(f"Could not find class folders in {directory}")
        if all(name.isdigit() for name in classes):
            classes = sorted(classes, key=int)
            return classes, {name: int(name) for name in classes}
        classes = sorted(classes)
        return classes, {name: index for index, name in enumerate(classes)}


class GaussianBlur:
    def __init__(self, radius_min: float = 0.1, radius_max: float = 2.0):
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, image):
        radius = random.uniform(self.radius_min, self.radius_max)
        return image.filter(ImageFilter.GaussianBlur(radius=radius))


class Solarization:
    def __call__(self, image):
        return ImageOps.solarize(image)


def create_three_augment(image_size: int, color_jitter: float = 0.3):
    secondary = [
        transforms.RandomChoice(
            [transforms.Grayscale(3), Solarization(), GaussianBlur()]
        )
    ]
    if color_jitter:
        secondary.append(transforms.ColorJitter(color_jitter, color_jitter, color_jitter))
    return transforms.Compose(
        [
            RandomResizedCropAndInterpolation(
                image_size, scale=(0.08, 1.0), interpolation="bicubic"
            ),
            transforms.RandomHorizontalFlip(),
            *secondary,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class DistributedEvalSampler(Sampler[int]):
    """Shard validation without padding or duplicated samples."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size


def _limited(dataset, limit: int):
    return Subset(dataset, range(min(limit, len(dataset)))) if limit > 0 else dataset


def create_loaders(args, cfg: ModelConfig, stage: str):
    root = Path(args.data_root)
    train_dir, val_dir = root / "train", root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Expected ImageNet train/ and val/ under {root}")

    if stage == "pretrain":
        train_transform = create_three_augment(cfg.image_size, args.color_jitter)
    else:
        train_transform = create_transform(
            input_size=(3, cfg.image_size, cfg.image_size),
            is_training=True,
            auto_augment=args.aa,
            interpolation="bicubic",
            color_jitter=args.color_jitter,
            re_prob=args.reprob,
            re_mode="pixel",
            re_count=1,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            use_prefetcher=False,
        )
    val_transform = create_transform(
        input_size=(3, cfg.image_size, cfg.image_size),
        is_training=False,
        interpolation="bicubic",
        crop_pct=1.0,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        use_prefetcher=False,
    )

    full_train = NumericImageFolder(train_dir, transform=train_transform)
    full_val = NumericImageFolder(val_dir, transform=val_transform)
    num_classes = max(full_train.class_to_idx.values()) + 1
    train_set = _limited(full_train, args.limit_train)
    val_set = _limited(full_val, args.limit_val)

    if stage == "pretrain":
        train_sampler = RepeatAugSampler(
            train_set,
            num_replicas=args.world_size,
            rank=args.rank,
            num_repeats=3,
            selected_round=256 if len(train_set) >= 256 else 0,
        )
        shuffle = False
    elif args.distributed:
        train_sampler = DistributedSampler(
            train_set, num_replicas=args.world_size, rank=args.rank, shuffle=True
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    val_sampler = (
        DistributedEvalSampler(val_set, args.rank, args.world_size)
        if args.distributed
        else None
    )
    common = dict(
        num_workers=args.workers,
        pin_memory=args.device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.validation_batch_size or args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, val_loader, train_sampler, num_classes


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _autocast(args):
    if not args.amp:
        return nullcontext()
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type=args.device.type, dtype=dtype)


def _reduce_sums(values: list[float], device: torch.device) -> list[float]:
    stats = torch.tensor(values, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return stats.tolist()


def create_mixup(args, num_classes: int) -> Optional[Mixup]:
    if args.mixup <= 0 and args.cutmix <= 0:
        return None
    return Mixup(
        mixup_alpha=args.mixup,
        cutmix_alpha=args.cutmix,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=args.smoothing,
        num_classes=num_classes,
    )


def create_train_loss(args, stage: str) -> nn.Module:
    if stage == "pretrain":
        return BinaryCrossEntropy(smoothing=0.0)
    if args.mixup > 0 or args.cutmix > 0:
        return SoftTargetCrossEntropy()
    if args.smoothing > 0:
        return LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    return nn.CrossEntropyLoss()


def train_one_epoch(
    args,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    mixup_fn: Optional[Mixup],
    model_ema: ModelEmaV3,
    loss_scaler: Optional[NativeScaler],
    num_updates: int,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = class_loss_sum = correct1 = correct5 = samples = 0.0
    batches = len(loader)
    for batch_index, (images, targets) in enumerate(loader):
        images = images.to(args.device, non_blocking=True)
        hard_targets = targets.to(args.device, non_blocking=True)
        targets = hard_targets
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        group_start = (batch_index // args.grad_accum_steps) * args.grad_accum_steps
        group_size = min(args.grad_accum_steps, batches - group_start)
        should_update = batch_index + 1 == group_start + group_size
        sync_context = nullcontext()
        if isinstance(model, DDP) and not should_update:
            sync_context = model.no_sync()

        with sync_context:
            with _autocast(args):
                output, _, recon = model(images)
                class_loss = criterion(output, targets)
                loss = class_loss
                if args.mode == "tied_data_rec":
                    loss = loss + args.lrec * recon[-1]
                backward_loss = loss / group_size
            if loss_scaler is not None:
                loss_scaler(
                    backward_loss,
                    optimizer,
                    clip_grad=args.clip_grad if should_update else None,
                    clip_mode="norm",
                    parameters=model.parameters(),
                    need_update=should_update,
                )
            else:
                backward_loss.backward()
                if should_update:
                    if args.clip_grad is not None:
                        dispatch_clip_grad(model.parameters(), args.clip_grad, mode="norm")
                    optimizer.step()

        batch_size = hard_targets.numel()
        loss_sum += loss.detach().item() * batch_size
        class_loss_sum += class_loss.detach().item() * batch_size
        predictions = output.detach()
        correct1 += predictions.argmax(1).eq(hard_targets).sum().item()
        k = min(5, predictions.shape[1])
        correct5 += predictions.topk(k, dim=1).indices.eq(hard_targets[:, None]).any(1).sum().item()
        samples += batch_size

        if should_update:
            optimizer.zero_grad(set_to_none=True)
            num_updates += 1
            model_ema.update(_unwrap(model), step=num_updates)

    loss_sum, class_loss_sum, correct1, correct5, samples = _reduce_sums(
        [loss_sum, class_loss_sum, correct1, correct5, samples], args.device
    )
    return {
        "loss": loss_sum / samples,
        "class_loss": class_loss_sum / samples,
        "acc1_hard": 100.0 * correct1 / samples,
        "acc5_hard": 100.0 * correct5 / samples,
    }, num_updates


@torch.no_grad()
def evaluate(args, model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = correct1 = correct5 = samples = 0.0
    for images, targets in loader:
        images = images.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)
        with _autocast(args):
            output = model(images)[0]
            loss = criterion(output, targets)
        batch_size = targets.numel()
        loss_sum += loss.item() * batch_size
        correct1 += output.argmax(1).eq(targets).sum().item()
        k = min(5, output.shape[1])
        correct5 += output.topk(k, dim=1).indices.eq(targets[:, None]).any(1).sum().item()
        samples += batch_size
    loss_sum, correct1, correct5, samples = _reduce_sums(
        [loss_sum, correct1, correct5, samples], args.device
    )
    return {
        "loss": loss_sum / samples,
        "acc1": 100.0 * correct1 / samples,
        "acc5": 100.0 * correct5 / samples,
    }


def _plain_args(args) -> dict[str, Any]:
    values = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            values[key] = str(value)
        elif isinstance(value, torch.device):
            values[key] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            values[key] = value
    return values


RESUME_TRAIN_CONFIG_KEYS = (
    "epochs",
    "opt",
    "lr",
    "weight_decay",
    "warmup_epochs",
    "warmup_lr",
    "min_lr",
    "batch_size",
    "validation_batch_size",
    "grad_accum_steps",
    "seed",
    "amp",
    "amp_dtype",
    "ema_decay",
    "mixup",
    "cutmix",
    "color_jitter",
    "smoothing",
    "reprob",
    "aa",
    "clip_grad",
    "lrec",
)


def training_signature(args, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        **{key: getattr(args, key) for key in RESUME_TRAIN_CONFIG_KEYS},
        "drop_path": args.drop_path,
    }


def capture_rng_state(args) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(args.device) if args.device.type == "cuda" else None,
    }


def collect_rng_states(args) -> Optional[list[dict[str, Any]]]:
    local_state = capture_rng_state(args)
    if not dist.is_initialized():
        return [local_state]
    gathered = [None] * args.world_size if is_primary(args) else None
    dist.gather_object(local_state, gathered, dst=0)
    return gathered


def checkpoint_state(
    args,
    stage: str,
    cfg: ModelConfig,
    model: nn.Module,
    model_ema: ModelEmaV3,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_scaler: Optional[NativeScaler],
    epoch: int,
    num_updates: int,
    best_metric: float,
    wandb_metadata: dict[str, Any],
    rng_by_rank: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "stage": stage,
        "epoch": epoch,
        "num_updates": num_updates,
        "best_metric": best_metric,
        "wandb": wandb_metadata,
        "model_config": asdict(cfg),
        "train_config": _plain_args(args),
        "training_signature": training_signature(args, stage),
        "model": _unwrap(model).state_dict(),
        "model_ema": model_ema.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
        "rng_world_size": args.world_size,
        "rng_by_rank": rng_by_rank,
    }


def atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def duplicate_checkpoint(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to overwrite permanent checkpoint {destination}")
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def save_epoch_checkpoints(
    args,
    state: dict[str, Any],
    epoch: int,
    improved: bool,
    final: bool = False,
) -> None:
    if not is_primary(args):
        return
    output_dir = Path(args.output_dir)
    latest = output_dir / "checkpoint_latest.pt"
    atomic_torch_save(state, latest)
    if improved:
        duplicate_checkpoint(latest, output_dir / "checkpoint_best.pt", overwrite=True)
    completed_epoch = epoch + 1
    if args.save_every > 0 and completed_epoch % args.save_every == 0:
        permanent = output_dir / f"checkpoint_epoch_{completed_epoch:04d}.pt"
        duplicate_checkpoint(latest, permanent, overwrite=False)
    if final:
        duplicate_checkpoint(latest, output_dir / "checkpoint_final.pt", overwrite=True)


def load_checkpoint_file(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} is not a state dictionary")
    return checkpoint


def expand_legacy_layerscale_state(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Expand legacy tied LayerScale vectors into one row per iteration."""
    target_state = model.state_dict()
    migrated = state_dict.copy()
    expanded_names: set[str] = set()
    for name, target in target_state.items():
        source = migrated.get(name)
        if (
            not name.startswith("block.ls_")
            or source is None
            or source.ndim + 1 != target.ndim
            or source.shape != target.shape[1:]
        ):
            continue
        migrated[name] = source.unsqueeze(0).expand(target.shape).clone()
        expanded_names.add(name)
    return migrated, expanded_names


def expand_legacy_layerscale_optimizer(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    expanded_names: set[str],
) -> None:
    """Expand optimizer moments for migrated per-iteration LayerScale tensors."""
    parameters = dict(model.named_parameters())
    for name in expanded_names:
        parameter = parameters[name]
        for key, value in list(optimizer.state[parameter].items()):
            if (
                torch.is_tensor(value)
                and value.ndim + 1 == parameter.ndim
                and value.shape == parameter.shape[1:]
            ):
                optimizer.state[parameter][key] = (
                    value.unsqueeze(0).expand(parameter.shape).clone()
                )


def restore_train_config(checkpoint: dict[str, Any], args) -> None:
    saved = checkpoint.get("train_config")
    if not isinstance(saved, dict):
        raise ValueError("Resume checkpoint does not contain train_config")
    missing = [key for key in RESUME_TRAIN_CONFIG_KEYS if key not in saved]
    if missing:
        raise ValueError(
            "Resume checkpoint is missing required training settings: "
            + ", ".join(missing)
        )
    for key in RESUME_TRAIN_CONFIG_KEYS:
        setattr(args, key, saved[key])


def _check_resume(checkpoint: dict[str, Any], args, stage: str, cfg: ModelConfig) -> None:
    if checkpoint.get("format_version") != 1 or checkpoint.get("stage") != stage:
        raise ValueError(f"{args.resume} is not a resumable {stage} checkpoint")
    saved_cfg = ModelConfig.from_dict(checkpoint["model_config"])
    if saved_cfg != cfg:
        raise ValueError("Model configuration differs from the resume checkpoint")


def restore_training_state(
    checkpoint: dict[str, Any],
    model: nn.Module,
    model_ema: ModelEmaV3,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_scaler: Optional[NativeScaler],
) -> tuple[int, int, float]:
    model_state, expanded_names = expand_legacy_layerscale_state(
        model, checkpoint["model"]
    )
    ema_state, _ = expand_legacy_layerscale_state(
        model_ema.module, checkpoint["model_ema"]
    )
    model.load_state_dict(model_state, strict=True)
    model_ema.module.load_state_dict(ema_state, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    expand_legacy_layerscale_optimizer(optimizer, model, expanded_names)
    if expanded_names:
        LOG.info(
            "Migrated legacy tied LayerScale checkpoint to per-iteration parameters: %s",
            sorted(expanded_names),
        )
    scheduler.load_state_dict(checkpoint["scheduler"])
    if loss_scaler is not None and checkpoint.get("scaler") is not None:
        loss_scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = int(checkpoint["epoch"]) + 1
    scheduler.step(start_epoch)
    return start_epoch, int(checkpoint.get("num_updates", 0)), float(checkpoint.get("best_metric", -1.0))


def restore_rng_state(checkpoint: dict[str, Any], args) -> None:
    states = checkpoint.get("rng_by_rank")
    if states is not None:
        saved_world_size = int(checkpoint.get("rng_world_size", len(states)))
        if saved_world_size != args.world_size or len(states) != args.world_size:
            raise ValueError(
                "Cannot restore per-rank RNG with a different world size: "
                f"checkpoint={saved_world_size}, current={args.world_size}"
            )
        state = states[args.rank]
    else:
        state = checkpoint.get("rng")
        if state is None:
            LOG.warning("Resume checkpoint has no RNG state; stochastic streams will restart")
            return
        if args.world_size != 1:
            LOG.warning(
                "Legacy checkpoint only contains rank-0 RNG state; skipping RNG restore for DDP"
            )
            return

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda")
    if args.device.type == "cuda" and cuda_state is not None:
        if isinstance(cuda_state, (list, tuple)):
            cuda_state = cuda_state[args.device.index or 0]
        torch.cuda.set_rng_state(cuda_state, device=args.device)


def load_initial_weights(model: nn.Module, checkpoint: dict[str, Any]) -> None:
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict):
        raise ValueError("Could not locate model weights in initial checkpoint")
    cleaned = {key.removeprefix("module."): value for key, value in state.items()}
    migrated, expanded_names = expand_legacy_layerscale_state(model, cleaned)
    model.load_state_dict(migrated, strict=True)
    if expanded_names:
        LOG.info(
            "Migrated legacy tied LayerScale weights to per-iteration parameters: %s",
            sorted(expanded_names),
        )


def _wandb_config(args, cfg: ModelConfig, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "model": asdict(cfg),
        "training": _plain_args(args),
        "effective_batch_size": args.batch_size * args.world_size * args.grad_accum_steps,
    }


def init_wandb(
    args,
    cfg: ModelConfig,
    stage: str,
    resume_checkpoint: Optional[dict[str, Any]],
):
    if not is_primary(args):
        return None

    saved = resume_checkpoint.get("wandb", {}) if resume_checkpoint is not None else {}
    run_id = saved.get("run_id") or args.wandb_run_id or None
    project = saved.get("project") or args.wandb_project
    entity = saved.get("entity") or args.wandb_entity or None
    name = saved.get("name") or args.wandb_name or f"{stage}-{cfg.mode}-seed{args.seed}"
    if args.resume and not run_id:
        LOG.warning(
            "Resume checkpoint has no W&B run ID; starting a new W&B run. "
            "Use --wandb-run-id to attach a legacy checkpoint to an existing run."
        )
    wandb_dir = Path(args.wandb_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    for env_name, child in (
        ("WANDB_CACHE_DIR", "cache"),
        ("WANDB_CONFIG_DIR", "config"),
        ("WANDB_DATA_DIR", "data"),
    ):
        directory = wandb_dir / child
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(env_name, str(directory))

    run = wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        resume="must" if args.resume and run_id else None,
        name=name,
        group=args.wandb_group or None,
        tags=args.wandb_tags or None,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        config=None if args.resume and run_id else _wandb_config(args, cfg, stage),
    )
    run.define_metric("epoch")
    run.define_metric("train/*", step_metric="epoch")
    run.define_metric("val/loss", step_metric="epoch", summary="min")
    run.define_metric("val/acc1", step_metric="epoch", summary="max")
    run.define_metric("val/acc5", step_metric="epoch", summary="max")
    run.define_metric("optimizer/*", step_metric="epoch")
    run.define_metric("time/*", step_metric="epoch")
    run.define_metric("diagnostics/*", step_metric="epoch")
    return run


def wandb_metadata(run) -> dict[str, Any]:
    if run is None:
        return {}
    return {
        "run_id": run.id,
        "project": run.project,
        "entity": run.entity,
        "name": run.name,
    }


def log_epoch_wandb(
    run,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    lr: float,
    num_updates: int,
    elapsed: float,
) -> None:
    if run is None:
        return
    run.log(
        {
            "epoch": epoch + 1,
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"val/{key}": value for key, value in val_metrics.items()},
            "optimizer/lr": lr,
            "optimizer/updates": num_updates,
            "time/epoch_seconds": elapsed,
        }
    )


def config_from_args(args, num_classes: int) -> ModelConfig:
    return ModelConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        dim=args.dim,
        n_reg=args.n_reg,
        heads=args.heads,
        steps=args.steps,
        num_classes=num_classes,
        mode=args.mode,
        attention=args.attention,
        func_k=args.func_k,
        func_lambda=args.func_lambda,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha if args.lora_alpha is not None else (args.lora_rank or 1.0),
        lora_compose=args.lora_compose,
        lora_dz=args.lora_dz,
        rmsnorm=args.rmsnorm,
        layerscale=args.layerscale,
        layerscale_init=args.layerscale_init,
        qk_rms_norm=args.qk_rms_norm,
        qk_rms_eps=args.qk_rms_eps,
        rope=args.rope,
        rope_base=args.rope_base,
        reg_rope=args.reg_rope,
        reg_rope_theta=args.reg_rope_theta,
        gamma_d=args.gamma_d,
        drop_path=args.drop_path,
    )


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=sorted(VALID_MODES), default="tied")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--n-reg", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--attention", choices=("softmax", "funcattn"), default="softmax")
    parser.add_argument("--func-k", type=int, default=64)
    parser.add_argument("--func-lambda", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=None)
    parser.add_argument("--lora-compose", choices=("indep", "cumul", "recur"), default="indep")
    parser.add_argument("--lora-dz", type=int, default=64)
    parser.add_argument("--rmsnorm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layerscale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--layerscale-init", type=float, default=1e-4)
    parser.add_argument("--qk-rms-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-rms-eps", type=float, default=1e-6)
    parser.add_argument("--rope", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rope-base", type=float, default=100.0)
    parser.add_argument("--reg-rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reg-rope-theta", type=float, default=10000.0)
    parser.add_argument("--gamma-d", type=float, default=0.5)
    parser.add_argument("--lrec", type=float, default=0.3)
    parser.add_argument("--drop-path", type=float, default=0.05)


def create_stage_parser(stage: str) -> argparse.ArgumentParser:
    if stage not in {"pretrain", "finetune"}:
        raise ValueError(stage)
    parser = argparse.ArgumentParser(description=f"PEQ ImageNet-1K {stage} with timm")
    add_model_arguments(parser)
    parser.add_argument("--data-root", default="/cis/home/cyang140/datasets/imagenet")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=512, help="batch size per GPU")
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dist-backend", default=None)
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--ema-decay", type=float, default=0.99996)
    parser.add_argument("--mixup", type=float, default=0.8)
    parser.add_argument("--cutmix", type=float, default=1.0)
    parser.add_argument("--color-jitter", type=float, default=0.3)
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get(
            "WANDB_PROJECT",
            "peq-imagenet-pretrain" if stage == "pretrain" else "peq-imagenet-finetune",
        ),
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME", ""))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP", ""))
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID", ""))
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", "wandb"))

    if stage == "pretrain":
        parser.add_argument("--initial-checkpoint", default="")
        parser.set_defaults(
            epochs=800,
            opt="lamb",
            lr=4e-3,
            weight_decay=0.05,
            warmup_epochs=5,
            warmup_lr=1e-6,
            min_lr=1e-5,
            smoothing=0.0,
            reprob=0.0,
            aa=None,
            clip_grad=1.0,
        )
    else:
        parser.add_argument("--pretrained-checkpoint", default="")
        parser.set_defaults(
            epochs=20,
            opt="adamw",
            lr=1e-5,
            weight_decay=0.1,
            warmup_epochs=5,
            warmup_lr=1e-6,
            min_lr=1e-6,
            smoothing=0.1,
            reprob=0.0,
            aa="rand-m9-mstd0.5-inc1",
            clip_grad=None,
        )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--opt")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--warmup-lr", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--smoothing", type=float)
    parser.add_argument("--reprob", type=float)
    parser.add_argument("--aa")
    parser.add_argument("--clip-grad", type=float)
    return parser


def apply_model_config(args, cfg: ModelConfig) -> None:
    mapping = {
        "image_size": "image_size",
        "patch_size": "patch_size",
        "dim": "dim",
        "n_reg": "n_reg",
        "heads": "heads",
        "steps": "steps",
        "mode": "mode",
        "attention": "attention",
        "func_k": "func_k",
        "func_lambda": "func_lambda",
        "lora_rank": "lora_rank",
        "lora_alpha": "lora_alpha",
        "lora_compose": "lora_compose",
        "lora_dz": "lora_dz",
        "rmsnorm": "rmsnorm",
        "layerscale": "layerscale",
        "layerscale_init": "layerscale_init",
        "qk_rms_norm": "qk_rms_norm",
        "qk_rms_eps": "qk_rms_eps",
        "rope": "rope",
        "rope_base": "rope_base",
        "reg_rope": "reg_rope",
        "reg_rope_theta": "reg_rope_theta",
        "gamma_d": "gamma_d",
        "drop_path": "drop_path",
    }
    for config_name, argument_name in mapping.items():
        setattr(args, argument_name, getattr(cfg, config_name))


def run_name_suffix(args, cfg: ModelConfig) -> str:
    return (
        f"lr{args.lr:g}_epoch{args.epochs}_BS{args.batch_size * args.world_size * args.grad_accum_steps}"
        f"_img{cfg.image_size}_D{cfg.dim}_T{cfg.steps}"
        f"_nreg{cfg.n_reg}_lorarank{cfg.lora_rank}"
    )


def append_run_suffix(value: str | Path, suffix: str) -> str:
    value = str(value)
    full_suffix = f"_{suffix}"
    return value if value.endswith(full_suffix) else value + full_suffix


def finalize_run_destinations(
    args,
    cfg: ModelConfig,
    stage: str,
    resume_checkpoint: Optional[dict[str, Any]],
) -> None:
    if args.resume:
        args.output_dir = str(Path(args.resume).parent)
        assert resume_checkpoint is not None
        saved_wandb = resume_checkpoint.get("wandb", {})
        saved_train_config = resume_checkpoint.get("train_config", {})
        saved_project = saved_wandb.get("project") or saved_train_config.get("wandb_project")
        if saved_project:
            args.wandb_project = saved_project
        return

    suffix = run_name_suffix(args, cfg)
    output_base = args.output_dir or f"outputs/peq_timm/{stage}/{cfg.mode}_seed{args.seed}"
    args.output_dir = append_run_suffix(output_base, suffix)
    args.wandb_project = append_run_suffix(args.wandb_project, suffix)


def run_stage(args, stage: str) -> None:
    setup_default_logging()
    args.device = init_distributed_device(args)
    if args.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if stage == "finetune" and not args.resume and not args.pretrained_checkpoint:
        raise ValueError("finetune requires --pretrained-checkpoint unless --resume is supplied")
    if stage == "pretrain" and args.resume and args.initial_checkpoint:
        raise ValueError("--resume and --initial-checkpoint are mutually exclusive")
    if args.resume and args.smoke:
        raise ValueError("--smoke cannot be combined with --resume")

    source = args.resume or (args.pretrained_checkpoint if stage == "finetune" else "")
    source_checkpoint = load_checkpoint_file(source) if source else None
    if args.resume:
        assert source_checkpoint is not None
        restore_train_config(source_checkpoint, args)
        if is_primary(args):
            LOG.info(
                "Restored training config from %s: epochs=%d lr=%g batch=%d "
                "grad_accum=%d mixup=%g cutmix=%g color_jitter=%g aa=%s reprob=%g",
                source,
                args.epochs,
                args.lr,
                args.batch_size,
                args.grad_accum_steps,
                args.mixup,
                args.cutmix,
                args.color_jitter,
                args.aa,
                args.reprob,
            )
    if source_checkpoint is not None:
        if "model_config" not in source_checkpoint:
            raise ValueError(f"Checkpoint {source} does not contain model_config")
        provisional_cfg = ModelConfig.from_dict(source_checkpoint["model_config"])
        apply_model_config(args, provisional_cfg)
    else:
        provisional_cfg = config_from_args(args, num_classes=1000)
    if args.amp and args.device.type not in {"cuda", "cpu"}:
        LOG.warning("Disabling AMP on device %s", args.device)
        args.amp = False
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be at least 1")
    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.limit_train = args.limit_train or max(
            args.batch_size * args.world_size * args.grad_accum_steps, 8
        )
        args.limit_val = args.limit_val or max(
            args.validation_batch_size or args.batch_size, 8
        )
        args.save_every = 1

    random_seed(args.seed, args.rank)
    train_loader, val_loader, train_sampler, num_classes = create_loaders(args, provisional_cfg, stage)
    if source_checkpoint is not None:
        cfg = provisional_cfg
        if cfg.num_classes != num_classes:
            raise ValueError(f"Checkpoint has {cfg.num_classes} classes, dataset has {num_classes}")
        if args.resume:
            _check_resume(source_checkpoint, args, stage, cfg)
    else:
        cfg = config_from_args(args, num_classes)

    finalize_run_destinations(
        args,
        cfg,
        stage,
        source_checkpoint if args.resume else None,
    )
    output_path = Path(args.output_dir)
    if not args.resume and (output_path / "checkpoint_latest.pt").exists():
        raise FileExistsError(
            f"{output_path} already contains checkpoint_latest.pt; use --resume or a new output directory"
        )
    if is_primary(args):
        output_path.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    model = Net(cfg).to(args.device)
    if stage == "pretrain" and args.initial_checkpoint:
        load_initial_weights(model, load_checkpoint_file(args.initial_checkpoint))
    if stage == "finetune" and not args.resume:
        if source_checkpoint is None or "model_ema" not in source_checkpoint:
            raise ValueError("Pretraining checkpoint does not contain model_ema")
        pretrained_state, expanded_names = expand_legacy_layerscale_state(
            model, source_checkpoint["model_ema"]
        )
        model.load_state_dict(pretrained_state, strict=True)
        if expanded_names:
            LOG.info(
                "Migrated legacy tied LayerScale pretraining weights to "
                "per-iteration parameters: %s",
                sorted(expanded_names),
            )

    optimizer = create_optimizer_v2(
        model,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler, _ = create_scheduler_v2(
        optimizer,
        sched="cosine",
        num_epochs=args.epochs,
        min_lr=args.min_lr,
        warmup_lr=args.warmup_lr,
        warmup_epochs=args.warmup_epochs,
        step_on_epochs=True,
    )
    scheduler.step(0)
    model_ema = ModelEmaV3(model, decay=args.ema_decay, foreach=args.device.type == "cuda")
    loss_scaler = (
        NativeScaler(device=args.device.type)
        if args.amp and args.amp_dtype == "float16" and args.device.type == "cuda"
        else None
    )
    start_epoch, num_updates, best_metric = 0, 0, -1.0
    if args.resume:
        assert source_checkpoint is not None
        start_epoch, num_updates, best_metric = restore_training_state(
            source_checkpoint, model, model_ema, optimizer, scheduler, loss_scaler
        )

    wandb_run = init_wandb(
        args,
        cfg,
        stage,
        source_checkpoint if args.resume else None,
    )

    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.device.index] if args.device.type == "cuda" else None,
            find_unused_parameters=cfg.mode not in DATA_MODES,
        )
    mixup_fn = create_mixup(args, num_classes)
    criterion = create_train_loss(args, stage)
    parameters_m = sum(parameter.numel() for parameter in _unwrap(model).parameters()) / 1e6
    effective_batch = args.batch_size * args.world_size * args.grad_accum_steps
    if is_primary(args):
        LOG.info(
            "stage=%s mode=%s params=%.2fM device=%s world=%d per_gpu_batch=%d effective_batch=%d",
            stage,
            cfg.mode,
            parameters_m,
            args.device,
            args.world_size,
            args.batch_size,
            effective_batch,
        )
        LOG.info("epochs=%d start_epoch=%d lr=%g output=%s", args.epochs, start_epoch, args.lr, args.output_dir)

    if args.resume:
        assert source_checkpoint is not None
        restore_rng_state(source_checkpoint, args)

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_metrics, num_updates = train_one_epoch(
            args,
            model,
            train_loader,
            optimizer,
            criterion,
            mixup_fn,
            model_ema,
            loss_scaler,
            num_updates,
        )
        val_metrics = evaluate(args, model, val_loader)
        improved = val_metrics["acc1"] > best_metric
        best_metric = max(best_metric, val_metrics["acc1"])
        scheduler.step(epoch + 1)
        elapsed = time.time() - epoch_start
        log_epoch_wandb(
            wandb_run,
            epoch,
            train_metrics,
            val_metrics,
            epoch_lr,
            num_updates,
            elapsed,
        )
        rng_by_rank = collect_rng_states(args)
        if is_primary(args):
            assert rng_by_rank is not None
            state = checkpoint_state(
                args,
                stage,
                cfg,
                model,
                model_ema,
                optimizer,
                scheduler,
                loss_scaler,
                epoch,
                num_updates,
                best_metric,
                wandb_metadata(wandb_run),
                rng_by_rank,
            )
            save_epoch_checkpoints(
                args,
                state,
                epoch,
                improved,
                final=epoch + 1 == args.epochs,
            )
        if is_primary(args):
            LOG.info(
                "epoch=%d/%d train_loss=%.4f val_loss=%.4f val_acc1=%.3f best=%.3f lr=%.3e sec=%.1f",
                epoch + 1,
                args.epochs,
                train_metrics["loss"],
                val_metrics["loss"],
                val_metrics["acc1"],
                best_metric,
                epoch_lr,
                elapsed,
            )

    if start_epoch >= args.epochs:
        if is_primary(args):
            LOG.info("Checkpoint already completed %d epochs; nothing to train", args.epochs)
    else:
        _unwrap(model).eval()
        with torch.no_grad(), _autocast(args):
            images, _ = next(iter(val_loader))
            _, resid, recon = _unwrap(model)(images.to(args.device, non_blocking=True), log=True)
        diagnostics = {"resid": resid, "recon": [value.item() for value in recon]}
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": args.epochs,
                    **{
                        f"diagnostics/resid_step_{index + 1}": value
                        for index, value in enumerate(diagnostics["resid"])
                    },
                    **{
                        f"diagnostics/recon_step_{index + 1}": value
                        for index, value in enumerate(diagnostics["recon"])
                    },
                }
            )

    if wandb_run is not None:
        wandb_run.finish()

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
