"""Fine-tune recurrent ConvNeXt V2 Atto from ImageNet-1K FCMAE weights.

The native Atto block depths are fixed at (2, 2, 6, 2). ``--stage-repeats``
controls how often each stage's complete block sequence is applied with shared
parameters; its downsample is always applied exactly once. The defaults match
the official ConvNeXt V2 Atto ImageNet-1K fine-tuning recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import shutil
import signal
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from safetensors.torch import load_file as load_safetensors
from timm.data import Mixup, create_transform
from timm.layers import drop_path
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEmaV3, NativeScaler, init_distributed_device, is_primary, random_seed, setup_default_logging
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

from imagenet_data import NumericImageFolder


LOG = logging.getLogger("convnextv2_atto_fcmae_finetune")
MODEL_ID = "convnextv2_atto.fcmae"
MODEL_FAMILY = "recurrent_convnextv2_atto_fcmae_finetune"
RECIPE_NAME = "convnextv2_atto_fcmae_imagenet1k_600e"
CHECKPOINT_FORMAT_VERSION = 1
ATTO_DEPTHS = (2, 2, 6, 2)
ATTO_DIMS = (40, 80, 160, 320)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OFFICIAL_RECIPE = {
    "image_size": 224,
    "epochs": 600,
    "warmup_epochs": 0,
    "base_lr": 2e-4,
    "reference_batch_size": 256,
    "global_batch_size": 1024,
    "min_lr": 1e-6,
    "layer_decay": 0.9,
    "weight_decay": 0.3,
    "drop_path_rate": 0.1,
    "reprob": 0.25,
    "mixup": 0.0,
    "cutmix": 0.0,
    "smoothing": 0.2,
    "aa": "rand-m9-mstd0.5-inc1",
    "ema_decay": 0.9999,
    "amp": True,
    "amp_dtype": "float16",
}

RESUME_ARGUMENT_KEYS = (
    "stage_repeats", "data_root", "output_dir", "image_size", "batch_size",
    "validation_batch_size", "max_global_batch_size", "grad_accum_steps",
    "workers", "limit_train",
    "limit_val", "epochs", "warmup_epochs", "base_lr", "reference_batch_size",
    "min_lr", "layer_decay", "weight_decay", "drop_path_rate", "reprob",
    "mixup", "cutmix", "smoothing", "aa", "ema_decay", "amp", "amp_dtype",
    "seed", "save_every", "smoke",
)

_STOP_REQUESTED = False
_LOGGING_CONFIGURED = False


def _request_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    LOG.warning("Received signal %s; checkpointing at the epoch boundary", signum)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_stop)


def setup_logging_once() -> None:
    global _LOGGING_CONFIGURED
    if not _LOGGING_CONFIGURED:
        setup_default_logging()
        _LOGGING_CONFIGURED = True


def parse_stage_repeats(value: str | tuple[int, ...] | list[int]) -> tuple[int, int, int, int]:
    values = value.split(",") if isinstance(value, str) else value
    try:
        repeats = tuple(int(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("stage_repeats must contain four comma-separated integers") from exc
    if len(repeats) != 4 or any(item < 1 for item in repeats):
        raise ValueError("stage_repeats must contain exactly four positive integers")
    return repeats  # type: ignore[return-value]


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-repeats", default="1,1,1,1")
    parser.add_argument("--data-root", default="/home/jhu/cyang140/scratch_abhatt40/cyang140/datasets/imagenet")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--pretrained-checkpoint", default="", help="local timm-format .safetensors/.pt FCMAE state dict")
    parser.add_argument("--resume", default="")
    parser.add_argument("--image-size", type=int, default=OFFICIAL_RECIPE["image_size"])
    parser.add_argument("--batch-size", type=int, default=32, help="per-rank batch size")
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument(
        "--max-global-batch-size",
        type=int,
        default=OFFICIAL_RECIPE["global_batch_size"],
        help="maximum effective global batch; accumulation is chosen automatically",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="optional compatibility override; must not exceed --max-global-batch-size",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=OFFICIAL_RECIPE["epochs"])
    parser.add_argument("--warmup-epochs", type=int, default=OFFICIAL_RECIPE["warmup_epochs"])
    parser.add_argument("--base-lr", type=float, default=OFFICIAL_RECIPE["base_lr"], help="base LR scaled from --reference-batch-size")
    parser.add_argument("--reference-batch-size", type=int, default=OFFICIAL_RECIPE["reference_batch_size"])
    parser.add_argument("--min-lr", type=float, default=OFFICIAL_RECIPE["min_lr"])
    parser.add_argument("--layer-decay", type=float, default=OFFICIAL_RECIPE["layer_decay"])
    parser.add_argument("--weight-decay", type=float, default=OFFICIAL_RECIPE["weight_decay"])
    parser.add_argument("--drop-path-rate", type=float, default=OFFICIAL_RECIPE["drop_path_rate"])
    parser.add_argument("--reprob", type=float, default=OFFICIAL_RECIPE["reprob"])
    parser.add_argument("--mixup", type=float, default=OFFICIAL_RECIPE["mixup"])
    parser.add_argument("--cutmix", type=float, default=OFFICIAL_RECIPE["cutmix"])
    parser.add_argument("--smoothing", type=float, default=OFFICIAL_RECIPE["smoothing"])
    parser.add_argument("--aa", default=OFFICIAL_RECIPE["aa"])
    parser.add_argument("--ema-decay", type=float, default=OFFICIAL_RECIPE["ema_decay"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dist-backend", default=None)
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=OFFICIAL_RECIPE["amp"])
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default=OFFICIAL_RECIPE["amp_dtype"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "convnextv2-atto-fcmae-finetune"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME", ""))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP", ""))
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID", ""))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=os.environ.get("WANDB_MODE", "disabled"))
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", "wandb/convnextv2-atto-fcmae"))
    return parser.parse_args(argv)


def expanded_drop_path_rates(repeats, maximum: float) -> tuple[float, ...]:
    repeats = parse_stage_repeats(repeats)
    applications = sum(depth * repeat for depth, repeat in zip(ATTO_DEPTHS, repeats))
    if not 0.0 <= maximum < 1.0:
        raise ValueError("drop_path_rate must be in [0, 1)")
    if applications == 1:
        return (0.0,)
    return tuple(maximum * index / (applications - 1) for index in range(applications))


def _forward_block(block: nn.Module, inputs: torch.Tensor, probability: float) -> torch.Tensor:
    """Run a timm ConvNeXt block with a per-invocation DropPath rate."""
    required = ("conv_dw", "norm", "mlp", "shortcut", "use_conv_mlp", "gamma")
    if any(not hasattr(block, name) for name in required):
        raise TypeError("Installed timm ConvNeXt block layout is incompatible")
    shortcut = inputs
    outputs = block.conv_dw(inputs)
    if block.use_conv_mlp:
        outputs = block.norm(outputs)
        outputs = block.mlp(outputs)
    else:
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = block.norm(outputs)
        outputs = block.mlp(outputs)
        outputs = outputs.permute(0, 3, 1, 2)
    if block.gamma is not None:
        outputs = outputs.mul(block.gamma.reshape(1, -1, 1, 1))
    outputs = drop_path(outputs, probability, block.training)
    return outputs + block.shortcut(shortcut)


class RecurrentConvNeXtV2Atto(nn.Module):
    """An exact timm Atto backbone with weight-tied whole-stage recurrence."""

    def __init__(self, base_model: nn.Module, stage_repeats=(1, 1, 1, 1), drop_path_rate=0.1):
        super().__init__()
        self.stage_repeats = parse_stage_repeats(stage_repeats)
        depths = tuple(len(stage.blocks) for stage in base_model.stages)
        dims = tuple(stage.blocks[0].conv_dw.out_channels for stage in base_model.stages)
        if depths != ATTO_DEPTHS or dims != ATTO_DIMS:
            raise ValueError(f"Expected Atto depths/dims {ATTO_DEPTHS}/{ATTO_DIMS}, got {depths}/{dims}")
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = float(drop_path_rate)
        self.drop_path_rates = expanded_drop_path_rates(self.stage_repeats, self.drop_path_rate)
        self.stem = base_model.stem
        self.stages = base_model.stages
        self.norm_pre = base_model.norm_pre
        self.head = base_model.head

    @property
    def block_applications(self) -> int:
        return len(self.drop_path_rates)

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.stem(inputs)
        offset = 0
        for stage, repeats, depth in zip(self.stages, self.stage_repeats, self.depths):
            outputs = stage.downsample(outputs)
            for _ in range(repeats):
                for block in stage.blocks:
                    outputs = _forward_block(block, outputs, self.drop_path_rates[offset])
                    offset += 1
        if offset != self.block_applications:
            raise RuntimeError("Internal DropPath schedule length mismatch")
        return self.norm_pre(outputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _local_pretrained_state(path: str | Path) -> Mapping[str, torch.Tensor]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix == ".safetensors":
        state: Any = load_safetensors(str(checkpoint_path), device="cpu")
    else:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(state, Mapping) and "state_dict" in state:
            state = state["state_dict"]
        elif isinstance(state, Mapping) and "model" in state:
            state = state["model"]
    if not isinstance(state, Mapping) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Local pretrained checkpoint must contain a timm-format tensor state dict")
    return state


def create_model(
    stage_repeats=(1, 1, 1, 1),
    drop_path_rate=0.1,
    num_classes=1000,
    *,
    pretrained=True,
    pretrained_checkpoint="",
) -> tuple[RecurrentConvNeXtV2Atto, dict[str, Any]]:
    if pretrained and pretrained_checkpoint:
        raise ValueError("pretrained=True and pretrained_checkpoint are mutually exclusive")
    try:
        base = timm.create_model(MODEL_ID, pretrained=pretrained, num_classes=0, drop_path_rate=0.0)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {MODEL_ID}; prefetch it or pass --pretrained-checkpoint"
        ) from exc
    source = "random-smoke"
    if pretrained_checkpoint:
        state = _local_pretrained_state(pretrained_checkpoint)
        result = base.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise ValueError(f"FCMAE state mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
        source = str(Path(pretrained_checkpoint).expanduser().resolve())
    elif pretrained:
        source = MODEL_ID
    backbone_state = base.state_dict()
    fingerprint = state_dict_sha256(backbone_state)
    base.reset_classifier(num_classes)
    nn.init.trunc_normal_(base.head.fc.weight, std=2e-5)
    nn.init.zeros_(base.head.fc.bias)
    model = RecurrentConvNeXtV2Atto(base, stage_repeats, drop_path_rate)
    metadata = {
        "model_id": MODEL_ID,
        "source": source,
        "state_sha256": fingerprint,
        "license": "CC-BY-NC-4.0",
        "pretrained": bool(pretrained or pretrained_checkpoint),
    }
    return model, metadata


def layer_id_for_parameter(name: str, depths=ATTO_DEPTHS) -> int:
    if name.startswith("stem."):
        return 1
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "stages":
        stage_id = int(parts[1])
        base = sum(depths[:stage_id]) + 1
        if parts[2] == "downsample":
            return base
        if len(parts) >= 5 and parts[2] == "blocks":
            return base + int(parts[3])
    return sum(depths) + 1


def parameter_groups(model: nn.Module, weight_decay: float, layer_decay: float):
    if layer_decay <= 0:
        raise ValueError("layer_decay must be positive")
    maximum_layer = sum(ATTO_DEPTHS) + 1
    groups: dict[tuple[int, bool], dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = parameter.ndim == 1 or name.endswith(".bias") or name.endswith((".gamma", ".beta"))
        layer_id = layer_id_for_parameter(name)
        key = (layer_id, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "weight_decay": 0.0 if no_decay else weight_decay,
                "lr_scale": layer_decay ** (maximum_layer - layer_id),
                "group_name": f"layer_{layer_id}_{'no_decay' if no_decay else 'decay'}",
            }
        groups[key]["params"].append(parameter)
    return list(groups.values())


def effective_batch_size(args) -> int:
    if args.grad_accum_steps is None:
        raise ValueError("grad_accum_steps has not been resolved")
    return args.batch_size * args.world_size * args.grad_accum_steps


def resolve_grad_accum_steps(args) -> int:
    """Choose the largest accumulation count within the global batch cap."""
    micro_global_batch = args.batch_size * args.world_size
    if args.max_global_batch_size < micro_global_batch:
        raise ValueError(
            f"max_global_batch_size={args.max_global_batch_size} is smaller than "
            f"batch_size*world_size={micro_global_batch}"
        )
    automatic = args.max_global_batch_size // micro_global_batch
    if args.grad_accum_steps is None:
        args.grad_accum_steps = automatic
    if args.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be positive")
    actual = micro_global_batch * args.grad_accum_steps
    if actual > args.max_global_batch_size:
        raise ValueError(
            f"Effective global batch {actual} exceeds max_global_batch_size="
            f"{args.max_global_batch_size}"
        )
    if args.grad_accum_steps != automatic:
        LOG.warning(
            "Using explicit grad_accum_steps=%d instead of automatic value %d",
            args.grad_accum_steps,
            automatic,
        )
    elif actual != args.max_global_batch_size:
        LOG.warning(
            "Maximum global batch %d is not divisible by micro global batch %d; "
            "using grad_accum_steps=%d and effective global batch %d",
            args.max_global_batch_size,
            micro_global_batch,
            args.grad_accum_steps,
            actual,
        )
    return args.grad_accum_steps


def scaled_peak_lr(args) -> float:
    return args.base_lr * effective_batch_size(args) / args.reference_batch_size


def create_optimizer(args, model: nn.Module):
    return torch.optim.AdamW(
        parameter_groups(model, args.weight_decay, args.layer_decay),
        lr=scaled_peak_lr(args), betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
    )


class UpdateCosineScheduler:
    def __init__(self, optimizer, peak_lr, min_lr, total_updates, warmup_updates=0):
        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.min_lr = float(min_lr)
        self.total_updates = int(total_updates)
        self.warmup_updates = int(warmup_updates)
        self.last_update = 0
        if self.total_updates < 1 or not 0 <= self.warmup_updates <= self.total_updates:
            raise ValueError("Invalid total/warmup update count")
        self.step_update(0)

    def base_lr_at(self, update: int) -> float:
        if self.warmup_updates and update < self.warmup_updates:
            return self.peak_lr * update / self.warmup_updates
        progress = (update - self.warmup_updates) / max(1, self.total_updates - self.warmup_updates)
        progress = min(1.0, max(0.0, progress))
        return self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def step_update(self, update: int) -> None:
        self.last_update = int(update)
        lr = self.base_lr_at(self.last_update)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)

    def state_dict(self) -> dict[str, Any]:
        return {"peak_lr": self.peak_lr, "min_lr": self.min_lr, "total_updates": self.total_updates,
                "warmup_updates": self.warmup_updates, "last_update": self.last_update}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = (self.peak_lr, self.min_lr, self.total_updates, self.warmup_updates)
        actual = (float(state["peak_lr"]), float(state["min_lr"]), int(state["total_updates"]), int(state["warmup_updates"]))
        if actual != expected:
            raise ValueError(f"Scheduler configuration mismatch: checkpoint={actual}, current={expected}")
        self.step_update(int(state["last_update"]))


class DistributedEvalSampler(Sampler[int]):
    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset, self.rank, self.world_size = dataset, rank, world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return max(0, (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size)


def _limited(dataset, limit: int):
    return Subset(dataset, range(min(limit, len(dataset)))) if limit > 0 else dataset


def create_loaders(args):
    root = Path(args.data_root)
    if not (root / "train").is_dir() or not (root / "val").is_dir():
        raise FileNotFoundError(f"Expected ImageNet train/ and val/ under {root}")
    train_transform = create_transform(
        input_size=(3, args.image_size, args.image_size), is_training=True,
        color_jitter=None, auto_augment=args.aa or None, interpolation="bicubic",
        re_prob=args.reprob, re_mode="pixel", re_count=1,
        mean=IMAGENET_MEAN, std=IMAGENET_STD, use_prefetcher=False,
    )
    val_transform = create_transform(
        input_size=(3, args.image_size, args.image_size), is_training=False,
        interpolation="bicubic", crop_pct=0.875, mean=IMAGENET_MEAN,
        std=IMAGENET_STD, use_prefetcher=False,
    )
    train_full = NumericImageFolder(root / "train", transform=train_transform)
    val_full = NumericImageFolder(root / "val", transform=val_transform)
    num_classes = max(train_full.class_to_idx.values()) + 1
    train_set, val_set = _limited(train_full, args.limit_train), _limited(val_full, args.limit_val)
    train_sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=args.rank, shuffle=True, seed=args.seed) if args.distributed else None
    val_sampler = DistributedEvalSampler(val_set, args.rank, args.world_size) if args.distributed else None
    common = dict(num_workers=args.workers, pin_memory=args.device.type == "cuda", persistent_workers=False)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, shuffle=train_sampler is None, drop_last=True, **common)
    val_loader = DataLoader(val_set, batch_size=args.validation_batch_size or args.batch_size, sampler=val_sampler, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, train_sampler, num_classes, len(train_set), len(val_set)


def create_mixup(args, num_classes: int) -> Optional[Mixup]:
    if args.mixup <= 0 and args.cutmix <= 0:
        return None
    return Mixup(
        mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, prob=1.0,
        switch_prob=0.5, mode="batch", label_smoothing=args.smoothing,
        num_classes=num_classes,
    )


def create_criterion(args) -> nn.Module:
    if args.mixup > 0 or args.cutmix > 0:
        return SoftTargetCrossEntropy()
    if args.smoothing > 0:
        return LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    return nn.CrossEntropyLoss()


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _autocast(args):
    if not args.amp:
        return nullcontext()
    dtype = torch.float16 if args.amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type=args.device.type, dtype=dtype)


def _reduce_sums(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def train_one_epoch(
    args, model, loader, optimizer, scheduler, criterion, mixup_fn,
    model_ema, loss_scaler, updates_per_epoch, num_updates,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = correct1 = correct5 = samples = 0.0
    required_batches = updates_per_epoch * args.grad_accum_steps
    consumed = 0
    for batch_index, (images, hard_targets) in enumerate(loader):
        if batch_index >= required_batches:
            break
        if batch_index % args.grad_accum_steps == 0:
            scheduler.step_update(num_updates)
        images = images.to(args.device, non_blocking=True)
        hard_targets = hard_targets.to(args.device, non_blocking=True)
        targets = hard_targets
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)
        should_update = (batch_index + 1) % args.grad_accum_steps == 0
        sync_context = model.no_sync() if isinstance(model, DDP) and not should_update else nullcontext()
        with sync_context:
            with _autocast(args):
                logits = model(images)
                loss = criterion(logits, targets)
                backward_loss = loss / args.grad_accum_steps
            if loss_scaler is not None:
                loss_scaler(backward_loss, optimizer, parameters=model.parameters(), need_update=should_update)
            else:
                backward_loss.backward()
                if should_update:
                    optimizer.step()
        batch_size = hard_targets.numel()
        loss_sum += loss.detach().item() * batch_size
        correct1 += logits.detach().argmax(1).eq(hard_targets).sum().item()
        correct5 += logits.detach().topk(min(5, logits.shape[1]), dim=1).indices.eq(hard_targets[:, None]).any(1).sum().item()
        samples += batch_size
        consumed += 1
        if should_update:
            optimizer.zero_grad(set_to_none=True)
            num_updates += 1
            model_ema.update(_unwrap(model), step=num_updates)
    if consumed != required_batches:
        raise RuntimeError(f"Train loader produced {consumed} micro-batches, expected {required_batches}")
    scheduler.step_update(num_updates)
    loss_sum, correct1, correct5, samples = _reduce_sums([loss_sum, correct1, correct5, samples], args.device)
    return {
        "loss": loss_sum / samples,
        "acc1_hard": 100.0 * correct1 / samples,
        "acc5_hard": 100.0 * correct5 / samples,
    }, num_updates


@torch.no_grad()
def evaluate(args, model, loader) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = correct1 = correct5 = samples = 0.0
    for images, targets in loader:
        images = images.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)
        with _autocast(args):
            logits = model(images)
            loss = criterion(logits, targets)
        batch_size = targets.numel()
        loss_sum += loss.item() * batch_size
        correct1 += logits.argmax(1).eq(targets).sum().item()
        correct5 += logits.topk(min(5, logits.shape[1]), dim=1).indices.eq(targets[:, None]).any(1).sum().item()
        samples += batch_size
    loss_sum, correct1, correct5, samples = _reduce_sums([loss_sum, correct1, correct5, samples], args.device)
    return {"loss": loss_sum / samples, "acc1": 100.0 * correct1 / samples, "acc5": 100.0 * correct5 / samples}


def capture_rng_state(args) -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(args.device) if args.device.type == "cuda" else None,
    }


def collect_rng_states(args):
    state = capture_rng_state(args)
    if not dist.is_initialized():
        return [state]
    gathered = [None] * args.world_size if is_primary(args) else None
    dist.gather_object(state, gathered, dst=0)
    return gathered


def restore_rng_state(args, checkpoint: Mapping[str, Any]) -> None:
    states = checkpoint.get("rng_by_rank")
    if not isinstance(states, list) or len(states) != args.world_size:
        raise ValueError("Resume checkpoint RNG state does not match current world size")
    state = states[args.rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if args.device.type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device=args.device)


def atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def atomic_json_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def duplicate_checkpoint(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def append_metric(path: Path, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def prepare_metrics_file(path: Path, start_epoch: int, resume: bool) -> None:
    if not resume:
        if path.exists():
            raise FileExistsError(f"Metrics file already exists: {path}")
        return
    retained = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if int(record["epoch"]) < start_epoch:
                    retained.append(record)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for record in retained:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _plain_args(args) -> dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        result[key] = str(value) if isinstance(value, (Path, torch.device)) else value
    return result


def recipe_mismatches(args) -> list[str]:
    actual = {
        "image_size": args.image_size, "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs, "base_lr": args.base_lr,
        "reference_batch_size": args.reference_batch_size,
        "global_batch_size": effective_batch_size(args), "min_lr": args.min_lr,
        "layer_decay": args.layer_decay, "weight_decay": args.weight_decay,
        "drop_path_rate": args.drop_path_rate, "reprob": args.reprob,
        "mixup": args.mixup, "cutmix": args.cutmix, "smoothing": args.smoothing,
        "aa": args.aa, "ema_decay": args.ema_decay, "amp": args.amp,
        "amp_dtype": args.amp_dtype,
    }
    mismatches = [f"{key}: expected={expected!r}, actual={actual[key]!r}" for key, expected in OFFICIAL_RECIPE.items() if actual[key] != expected]
    if args.limit_train:
        mismatches.append(f"limit_train: expected=0, actual={args.limit_train}")
    if args.limit_val:
        mismatches.append(f"limit_val: expected=0, actual={args.limit_val}")
    return mismatches


def default_output_dir(args, repeats) -> str:
    repeat_slug = "-".join(map(str, repeats))
    return f"outputs/convnextv2_atto_fcmae_repeats-{repeat_slug}_ep{args.epochs}_gbs{effective_batch_size(args)}_lr{scaled_peak_lr(args):g}_seed{args.seed}"


def build_config(args, model, source_metadata, num_classes, train_size, val_size, updates_per_epoch):
    mismatches = recipe_mismatches(args)
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_family": MODEL_FAMILY,
        "recipe": RECIPE_NAME,
        "training_recipe_exact": not mismatches and not args.smoke,
        "training_recipe_mismatches": mismatches,
        "pretrained": source_metadata,
        "architecture": {
            "depths": list(model.depths), "dims": list(model.dims),
            "stage_repeats": list(model.stage_repeats),
            "unique_blocks": sum(model.depths), "block_applications": model.block_applications,
            "downsample_applications": [1, 1, 1, 1], "weight_tied_stage_recurrence": True,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "drop_path_rate": model.drop_path_rate,
            "drop_path_schedule": "expanded_global_linear",
            "drop_path_rates": list(model.drop_path_rates),
        },
        "training": {
            **_plain_args(args), "optimizer": "adamw", "optimizer_betas": [0.9, 0.999],
            "optimizer_eps": 1e-8, "layer_decay_type": "single",
            "peak_lr": scaled_peak_lr(args), "effective_batch_size": effective_batch_size(args),
            "updates_per_epoch": updates_per_epoch, "total_updates": args.epochs * updates_per_epoch,
            "gradient_clip": None, "random_erasing_mode": "pixel",
            "random_erasing_count": 1, "train_interpolation": "bicubic",
            "validation_crop_pct": 0.875,
        },
        "dataset": {"num_classes": num_classes, "train_size": train_size, "val_size": val_size},
    }


def init_wandb(args, config, checkpoint):
    if not is_primary(args) or args.wandb_mode == "disabled":
        return None
    saved = checkpoint.get("wandb", {}) if checkpoint else {}
    run_id = saved.get("run_id") or args.wandb_run_id or None
    Path(args.wandb_dir).mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=saved.get("project") or args.wandb_project,
        entity=saved.get("entity") or args.wandb_entity or None,
        name=saved.get("name") or args.wandb_name or None,
        group=saved.get("group") or args.wandb_group or None,
        id=run_id, resume="allow" if run_id else None, mode=args.wandb_mode,
        dir=args.wandb_dir, config=None if checkpoint and run_id else config,
    )


def wandb_metadata(run) -> dict[str, Any]:
    if run is None:
        return {}
    return {"run_id": run.id, "project": run.project, "entity": run.entity, "name": run.name, "group": run.group}


def checkpoint_state(args, config, model, model_ema, optimizer, scheduler, loss_scaler,
                     epoch, num_updates, best_ema_acc1, best_raw_acc1, rng_by_rank, run):
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION, "model_family": MODEL_FAMILY,
        "epoch": epoch, "num_updates": num_updates,
        "best_ema_acc1": best_ema_acc1, "best_raw_acc1": best_raw_acc1,
        "world_size": args.world_size, "arguments": _plain_args(args), "config": config,
        "model": _unwrap(model).state_dict(), "model_ema": model_ema.module.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "scaler": loss_scaler.state_dict() if loss_scaler else None,
        "rng_by_rank": rng_by_rank, "wandb": wandb_metadata(run),
    }


def load_resume(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION or checkpoint.get("model_family") != MODEL_FAMILY:
        raise ValueError(f"Incompatible resume checkpoint: {checkpoint_path}")
    return checkpoint


def restore_resume_arguments(args, checkpoint) -> None:
    saved = checkpoint.get("arguments")
    if not isinstance(saved, Mapping):
        raise ValueError("Resume checkpoint has no arguments dictionary")
    saved = dict(saved)
    saved.setdefault(
        "max_global_batch_size",
        int(saved.get("batch_size", 0))
        * int(checkpoint.get("world_size", 1))
        * int(saved.get("grad_accum_steps", 0)),
    )
    missing = [key for key in RESUME_ARGUMENT_KEYS if key not in saved]
    if missing:
        raise ValueError("Resume checkpoint is missing arguments: " + ", ".join(missing))
    for key in RESUME_ARGUMENT_KEYS:
        setattr(args, key, saved[key])


def _apply_smoke_defaults(args) -> None:
    args.epochs = 1
    args.warmup_epochs = 0
    args.batch_size = min(args.batch_size, 2)
    args.validation_batch_size = min(args.validation_batch_size or args.batch_size, 2)
    args.max_global_batch_size = args.batch_size * args.world_size
    args.grad_accum_steps = None
    args.workers = 0
    args.limit_train = args.limit_train or 4
    args.limit_val = args.limit_val or 2
    args.save_every = 1


def synchronized_stop_requested(args) -> bool:
    requested = torch.tensor(int(_STOP_REQUESTED), device=args.device, dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(requested, op=dist.ReduceOp.MAX)
    return bool(requested.item())


def _validate_args(args) -> None:
    if min(args.batch_size, args.max_global_batch_size, args.reference_batch_size, args.epochs) < 1:
        raise ValueError("batch sizes, reference batch, and epochs must be positive")
    if not 0 <= args.warmup_epochs <= args.epochs:
        raise ValueError("warmup_epochs must be in [0, epochs]")
    if min(args.base_lr, args.min_lr, args.weight_decay) < 0 or args.layer_decay <= 0:
        raise ValueError("learning rates/weight decay must be non-negative and layer_decay positive")
    if not 0 <= args.reprob <= 1 or not 0 <= args.smoothing < 1:
        raise ValueError("reprob must be in [0,1] and smoothing in [0,1)")
    if args.mixup < 0 or args.cutmix < 0 or not 0 <= args.ema_decay < 1:
        raise ValueError("mixup/cutmix must be non-negative and ema_decay in [0,1)")
    if args.save_every < 0:
        raise ValueError("save_every must be non-negative")


def run(args) -> None:
    setup_logging_once()
    install_signal_handlers()
    args.device = init_distributed_device(args)
    if args.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if args.resume and args.smoke:
        raise ValueError("--resume and --smoke are mutually exclusive")
    if args.resume and args.pretrained_checkpoint:
        raise ValueError("--resume and --pretrained-checkpoint are mutually exclusive")
    resume_checkpoint = load_resume(args.resume) if args.resume else None
    if resume_checkpoint:
        if int(resume_checkpoint.get("world_size", -1)) != args.world_size:
            raise ValueError("Resume world size differs from the checkpoint")
        restore_resume_arguments(args, resume_checkpoint)
    if args.smoke:
        _apply_smoke_defaults(args)
    repeats = parse_stage_repeats(args.stage_repeats)
    args.stage_repeats = ",".join(map(str, repeats))
    if args.device.type != "cuda" and args.amp:
        LOG.warning("Disabling AMP on non-CUDA device %s", args.device)
        args.amp = False
    resolve_grad_accum_steps(args)
    _validate_args(args)
    random_seed(args.seed, args.rank)
    train_loader, val_loader, train_sampler, num_classes, train_size, val_size = create_loaders(args)
    if num_classes != 1000 and not args.smoke:
        raise ValueError(f"Expected 1000 ImageNet classes, found {num_classes}")
    updates_per_epoch = train_size // effective_batch_size(args)
    if updates_per_epoch < 1:
        raise ValueError(f"Training set size {train_size} is smaller than effective batch {effective_batch_size(args)}")
    if len(train_loader) < updates_per_epoch * args.grad_accum_steps:
        raise ValueError("Train loader cannot provide the required optimizer updates")

    if args.distributed and not args.resume and not args.pretrained_checkpoint and not args.smoke:
        if is_primary(args):
            create_model(repeats, args.drop_path_rate, num_classes, pretrained=True)
        dist.barrier()
    model, source_metadata = create_model(
        repeats, args.drop_path_rate, num_classes,
        pretrained=not args.resume and not args.pretrained_checkpoint and not args.smoke,
        pretrained_checkpoint=args.pretrained_checkpoint if not args.resume else "",
    )
    if resume_checkpoint:
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        source_metadata = resume_checkpoint["config"]["pretrained"]
    model.to(args.device)
    args.output_dir = args.output_dir or default_output_dir(args, repeats)
    args.wandb_name = args.wandb_name or f"convnextv2-atto-fcmae-r{'-'.join(map(str, repeats))}-ep{args.epochs}"
    output_dir = Path(args.output_dir)
    latest_path = output_dir / "checkpoint_latest.pt"
    if not args.resume and latest_path.exists():
        raise FileExistsError(f"{latest_path} exists; use --resume or a new output directory")

    config = build_config(args, model, source_metadata, num_classes, train_size, val_size, updates_per_epoch)
    if is_primary(args):
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_save(config, output_dir / "config.json")
        for mismatch in config["training_recipe_mismatches"]:
            LOG.warning("Recipe override: %s", mismatch)
    if dist.is_initialized():
        dist.barrier()

    optimizer = create_optimizer(args, model)
    scheduler = UpdateCosineScheduler(
        optimizer, scaled_peak_lr(args), args.min_lr,
        args.epochs * updates_per_epoch, args.warmup_epochs * updates_per_epoch,
    )
    if resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        scheduler.load_state_dict(resume_checkpoint["scheduler"])
    if args.distributed:
        model = DDP(model, device_ids=[args.device.index] if args.device.type == "cuda" else None, find_unused_parameters=False)
    model_ema = ModelEmaV3(_unwrap(model), decay=args.ema_decay, use_warmup=False, foreach=args.device.type == "cuda")
    loss_scaler = NativeScaler(device=args.device.type) if args.amp and args.amp_dtype == "float16" and args.device.type == "cuda" else None
    start_epoch = num_updates = 0
    best_ema_acc1 = best_raw_acc1 = -1.0
    if resume_checkpoint:
        model_ema.module.load_state_dict(resume_checkpoint["model_ema"], strict=True)
        if loss_scaler and resume_checkpoint.get("scaler") is not None:
            loss_scaler.load_state_dict(resume_checkpoint["scaler"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        num_updates = int(resume_checkpoint["num_updates"])
        best_ema_acc1 = float(resume_checkpoint.get("best_ema_acc1", -1.0))
        best_raw_acc1 = float(resume_checkpoint.get("best_raw_acc1", -1.0))
        scheduler.step_update(num_updates)

    mixup_fn = create_mixup(args, num_classes)
    criterion = create_criterion(args)
    run_handle = init_wandb(args, config, resume_checkpoint)
    metrics_path = output_dir / "metrics.jsonl"
    if is_primary(args):
        prepare_metrics_file(metrics_path, start_epoch, bool(args.resume))
        LOG.info("model=%s repeats=%s params=%d block_applications=%d source=%s hash=%s",
                 MODEL_ID, repeats, config["architecture"]["parameters"], model_ema.module.block_applications,
                 source_metadata["source"], source_metadata["state_sha256"])
        LOG.info("effective_batch=%d peak_lr=%g updates_per_epoch=%d recipe_exact=%s output=%s",
                 effective_batch_size(args), scaled_peak_lr(args), updates_per_epoch,
                 config["training_recipe_exact"], output_dir)
    if resume_checkpoint:
        restore_rng_state(args, resume_checkpoint)

    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_metrics, num_updates = train_one_epoch(
            args, model, train_loader, optimizer, scheduler, criterion, mixup_fn,
            model_ema, loss_scaler, updates_per_epoch, num_updates,
        )
        raw_metrics = evaluate(args, model, val_loader)
        ema_metrics = evaluate(args, model_ema.module, val_loader)
        improved_raw = raw_metrics["acc1"] > best_raw_acc1
        improved_ema = ema_metrics["acc1"] > best_ema_acc1
        best_raw_acc1 = max(best_raw_acc1, raw_metrics["acc1"])
        best_ema_acc1 = max(best_ema_acc1, ema_metrics["acc1"])
        record = {
            "epoch": epoch, "num_updates": num_updates,
            "lr": max(group["lr"] for group in optimizer.param_groups),
            "train": train_metrics, "raw": raw_metrics, "ema": ema_metrics,
            "best_raw_acc1": best_raw_acc1, "best_ema_acc1": best_ema_acc1,
            "epoch_sec": time.time() - started,
        }
        if is_primary(args):
            append_metric(metrics_path, record)
            if run_handle:
                run_handle.log(record, step=epoch + 1)
        rng_by_rank = collect_rng_states(args)
        if is_primary(args):
            state = checkpoint_state(
                args, config, model, model_ema, optimizer, scheduler, loss_scaler,
                epoch, num_updates, best_ema_acc1, best_raw_acc1, rng_by_rank, run_handle,
            )
            atomic_torch_save(state, latest_path)
            if improved_ema:
                duplicate_checkpoint(latest_path, output_dir / "checkpoint_best.pt")
            if improved_raw:
                duplicate_checkpoint(latest_path, output_dir / "checkpoint_best_raw.pt")
            if args.save_every and (epoch + 1) % args.save_every == 0:
                duplicate_checkpoint(latest_path, output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt")
            if epoch + 1 == args.epochs:
                duplicate_checkpoint(latest_path, output_dir / "checkpoint_final.pt")
            LOG.info("epoch=%d/%d loss=%.4f raw=%.3f ema=%.3f best_ema=%.3f sec=%.1f",
                     epoch + 1, args.epochs, train_metrics["loss"], raw_metrics["acc1"],
                     ema_metrics["acc1"], best_ema_acc1, record["epoch_sec"])
        if synchronized_stop_requested(args):
            break
    if run_handle:
        run_handle.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
