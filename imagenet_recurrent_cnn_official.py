"""ConvNeXt ImageNet-1K training with the official 300-epoch recipe.

The model family and ARR1/ARR2 architecture syntax are shared with
``recurrent_cnn.py``.  The default architecture is the project's recurrent
ConvNeXt V2, while the optimization and augmentation defaults reproduce Table 5
of "A ConvNet for the 2020s".
"""

from __future__ import annotations

import argparse
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
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, Subset
from torch.utils.data.distributed import DistributedSampler

from timm.data import Mixup, create_transform
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
from timm.utils import (
    ModelEmaV3,
    NativeScaler,
    init_distributed_device,
    is_primary,
    random_seed,
    setup_default_logging,
)

from imagenet_data import NumericImageFolder
from recurrent_cnn import (
    DELTA_BACKEND,
    DELTA_CHUNK_SIZE,
    DELTA_CONV_SIZE,
    DELTA_NORM_EPS,
    DEFAULT_N_REG,
    DEFAULT_REG_MODE,
    REG_HEADS,
    REG_SDPA_BACKEND,
    RecurrentCNN,
    model_arch_name,
    validate_register_arrays,
    validate_stage_arrays,
)


LOG = logging.getLogger("recurrent_convnext_official")
CHECKPOINT_FORMAT_VERSION = 1
MODEL_FAMILY = "recurrent_cnn_official"
RECIPE_NAME = "convnext_v1_imagenet1k_300e_table5"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NATIVE_TINY_DEPTHS = (3, 3, 9, 3)
NATIVE_TINY_REPEATS = (1, 1, 1, 1)

OFFICIAL_RECIPE = {
    "image_size": 224,
    "epochs": 300,
    "warmup_epochs": 20,
    "base_lr": 4e-3,
    "reference_batch_size": 4096,
    "global_batch_size": 4096,
    "warmup_lr": 1e-6,
    "min_lr": 1e-6,
    "weight_decay": 0.05,
    "mixup": 0.8,
    "cutmix": 1.0,
    "smoothing": 0.1,
    "reprob": 0.25,
    "aa": "rand-m9-mstd0.5-inc1",
    "color_jitter": 0.4,
    "drop_path_rate": 0.1,
    "ema_decay": 0.9999,
}

RESUME_ARGUMENT_KEYS = (
    "arr1",
    "arr2",
    "reg_mode",
    "n_reg",
    "delta_mode",
    "reg_head",
    "convnext_version",
    "drop_path_rate",
    "image_size",
    "epochs",
    "warmup_epochs",
    "base_lr",
    "reference_batch_size",
    "warmup_lr",
    "min_lr",
    "weight_decay",
    "mixup",
    "cutmix",
    "smoothing",
    "reprob",
    "aa",
    "color_jitter",
    "ema_decay",
    "batch_size",
    "grad_accum_steps",
    "seed",
    "amp",
    "amp_dtype",
    "strict_official_recipe",
)

_STOP_REQUESTED = False
_LOGGING_CONFIGURED = False


def _request_stop(signum, _frame) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    LOG.warning("Received signal %s; will checkpoint and exit at the epoch boundary", signum)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _request_stop)


def setup_logging_once() -> None:
    global _LOGGING_CONFIGURED
    if not _LOGGING_CONFIGURED:
        setup_default_logging()
        _LOGGING_CONFIGURED = True


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description="Recurrent ConvNeXt ImageNet-1K training with the official 300-epoch recipe"
    )
    parser.add_argument("--arr1", default="1,1,2,0")
    parser.add_argument("--arr2", default="3,3,6,0")
    parser.add_argument("--reg-mode", default="0,0,0,0")
    parser.add_argument("--n-reg", default="8,8,8,8")
    parser.add_argument(
        "--delta-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply one patch DeltaNet residual before RATS in every enabled register stage",
    )
    parser.add_argument(
        "--reg-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="concatenate final-stage register mean with pooled feature tokens",
    )
    parser.add_argument("--convnext-version", type=int, choices=(1, 2), default=2)
    parser.add_argument("--drop-path-rate", type=float, default=OFFICIAL_RECIPE["drop_path_rate"])

    parser.add_argument("--data-root", default="/cis/project/peq_project/imagenet-1k")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--image-size", type=int, default=OFFICIAL_RECIPE["image_size"])
    parser.add_argument("--batch-size", type=int, default=128, help="per-rank batch size")
    parser.add_argument("--validation-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)

    parser.add_argument(
        "--epochs",
        type=int,
        default=os.environ.get("EPOCHS", OFFICIAL_RECIPE["epochs"]),
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=os.environ.get("WARMUP_EPOCHS", OFFICIAL_RECIPE["warmup_epochs"]),
    )
    parser.add_argument("--base-lr", type=float, default=OFFICIAL_RECIPE["base_lr"])
    parser.add_argument(
        "--reference-batch-size",
        type=int,
        default=OFFICIAL_RECIPE["reference_batch_size"],
    )
    parser.add_argument("--warmup-lr", type=float, default=OFFICIAL_RECIPE["warmup_lr"])
    parser.add_argument("--min-lr", type=float, default=OFFICIAL_RECIPE["min_lr"])
    parser.add_argument("--weight-decay", type=float, default=OFFICIAL_RECIPE["weight_decay"])
    parser.add_argument("--mixup", type=float, default=OFFICIAL_RECIPE["mixup"])
    parser.add_argument("--cutmix", type=float, default=OFFICIAL_RECIPE["cutmix"])
    parser.add_argument("--smoothing", type=float, default=OFFICIAL_RECIPE["smoothing"])
    parser.add_argument("--reprob", type=float, default=OFFICIAL_RECIPE["reprob"])
    parser.add_argument("--aa", default=OFFICIAL_RECIPE["aa"])
    parser.add_argument("--color-jitter", type=float, default=OFFICIAL_RECIPE["color_jitter"])
    parser.add_argument("--ema-decay", type=float, default=OFFICIAL_RECIPE["ema_decay"])
    parser.add_argument(
        "--strict-official-recipe",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dist-backend", default=None)
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--smoke", action="store_true")

    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "recurrent-convnext-imagenet"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME", ""))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP", ""))
    parser.add_argument("--wandb-run-id", default=os.environ.get("WANDB_RUN_ID", ""))
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("WANDB_MODE", "disabled"),
    )
    parser.add_argument("--wandb-dir", default=os.environ.get("WANDB_DIR", "wandb/recurrent-convnext-official"))
    args = parser.parse_args(argv)
    if args.strict_official_recipe is None:
        args.strict_official_recipe = (
            args.epochs == OFFICIAL_RECIPE["epochs"]
            and args.warmup_epochs == OFFICIAL_RECIPE["warmup_epochs"]
        )
    return args


def _plain_args(args) -> dict[str, Any]:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, torch.device):
            result[key] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            result[key] = value
    return result


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {path} is not a dictionary")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint {path} has incompatible format_version="
            f"{checkpoint.get('format_version')!r}"
        )
    if checkpoint.get("model_family") != MODEL_FAMILY:
        raise ValueError(
            f"Checkpoint {path} has model_family={checkpoint.get('model_family')!r}; "
            f"expected {MODEL_FAMILY!r}"
        )
    return checkpoint


def _restore_resume_arguments(args, checkpoint: dict[str, Any]) -> None:
    saved = checkpoint.get("arguments")
    if not isinstance(saved, dict):
        raise ValueError("Resume checkpoint has no arguments dictionary")
    saved = dict(saved)
    saved.setdefault("reg_mode", ",".join(map(str, DEFAULT_REG_MODE)))
    saved.setdefault("n_reg", ",".join(map(str, DEFAULT_N_REG)))
    saved.setdefault("delta_mode", False)
    saved.setdefault("reg_head", False)
    missing = [key for key in RESUME_ARGUMENT_KEYS if key not in saved]
    if missing:
        raise ValueError("Resume checkpoint is missing arguments: " + ", ".join(missing))
    mismatches = [
        f"{key}: checkpoint={saved[key]!r}, requested={getattr(args, key)!r}"
        for key in RESUME_ARGUMENT_KEYS
        if getattr(args, key) != saved[key]
    ]
    if mismatches:
        raise ValueError(
            "Resume arguments differ from the checkpoint:\n  "
            + "\n  ".join(mismatches)
        )
    for key in RESUME_ARGUMENT_KEYS:
        setattr(args, key, saved[key])
    args.output_dir = str(Path(args.resume).resolve().parent)


def _apply_smoke_defaults(args) -> None:
    args.epochs = 1
    args.warmup_epochs = 0
    args.batch_size = min(args.batch_size, 2)
    args.validation_batch_size = min(args.validation_batch_size or args.batch_size, 2)
    args.grad_accum_steps = 1
    args.workers = 0
    args.limit_train = args.limit_train or max(2 * args.batch_size * args.world_size, 4)
    args.limit_val = args.limit_val or max(args.validation_batch_size * args.world_size, 2)
    args.save_every = 1


def effective_batch_size(args) -> int:
    return args.batch_size * args.world_size * args.grad_accum_steps


def scaled_peak_lr(args) -> float:
    return args.base_lr * effective_batch_size(args) / args.reference_batch_size


def official_recipe_mismatches(args) -> list[str]:
    actual = {
        "image_size": args.image_size,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "base_lr": args.base_lr,
        "reference_batch_size": args.reference_batch_size,
        "global_batch_size": effective_batch_size(args),
        "warmup_lr": args.warmup_lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "mixup": args.mixup,
        "cutmix": args.cutmix,
        "smoothing": args.smoothing,
        "reprob": args.reprob,
        "aa": args.aa,
        "color_jitter": args.color_jitter,
        "drop_path_rate": args.drop_path_rate,
        "ema_decay": args.ema_decay,
    }
    mismatches = [
        f"{key}: expected={expected!r}, actual={actual[key]!r}"
        for key, expected in OFFICIAL_RECIPE.items()
        if actual[key] != expected
    ]
    if args.limit_train != 0:
        mismatches.append(f"limit_train: expected=0, actual={args.limit_train}")
    if args.limit_val != 0:
        mismatches.append(f"limit_val: expected=0, actual={args.limit_val}")
    return mismatches


class DistributedEvalSampler(Sampler[int]):
    """Shard validation data without the padding used by DistributedSampler."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return max(0, (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size)


def _limited(dataset, limit: int):
    return Subset(dataset, range(min(limit, len(dataset)))) if limit > 0 else dataset


def create_loaders(args):
    root = Path(args.data_root)
    train_dir, val_dir = root / "train", root / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(f"Expected ImageNet train/ and val/ under {root}")

    train_transform = create_transform(
        input_size=(3, args.image_size, args.image_size),
        is_training=True,
        auto_augment=args.aa or None,
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
        input_size=(3, args.image_size, args.image_size),
        is_training=False,
        interpolation="bicubic",
        crop_pct=0.875,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        use_prefetcher=False,
    )
    full_train = NumericImageFolder(train_dir, transform=train_transform)
    full_val = NumericImageFolder(val_dir, transform=val_transform)
    num_classes = max(full_train.class_to_idx.values()) + 1
    train_set = _limited(full_train, args.limit_train)
    val_set = _limited(full_val, args.limit_val)

    train_sampler = (
        DistributedSampler(
            train_set,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=args.seed,
        )
        if args.distributed
        else None
    )
    val_sampler = (
        DistributedEvalSampler(val_set, args.rank, args.world_size)
        if args.distributed
        else None
    )
    common = {
        "num_workers": args.workers,
        "pin_memory": args.device.type == "cuda",
        # Recreating workers at each epoch makes epoch-boundary RNG restoration
        # exact; persistent worker RNG streams cannot be represented in a
        # conventional trainer checkpoint.
        "persistent_workers": False,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
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
    return train_loader, val_loader, train_sampler, num_classes, len(train_set), len(val_set)


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


def create_train_criterion(args) -> nn.Module:
    if args.mixup > 0 or args.cutmix > 0:
        return SoftTargetCrossEntropy()
    if args.smoothing > 0:
        return LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    return nn.CrossEntropyLoss()


def create_optimizer(args, model: nn.Module):
    return create_optimizer_v2(
        model,
        opt="adamw",
        lr=scaled_peak_lr(args),
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        filter_bias_and_bn=True,
    )


def create_update_scheduler(args, optimizer, updates_per_epoch: int):
    warmup_updates = args.warmup_epochs * updates_per_epoch
    decay_updates = max(1, (args.epochs - args.warmup_epochs) * updates_per_epoch)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=decay_updates,
        lr_min=args.min_lr,
        warmup_t=warmup_updates,
        warmup_lr_init=args.warmup_lr,
        warmup_prefix=True,
        t_in_epochs=False,
    )
    scheduler.step_update(0)
    return scheduler


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _autocast(args):
    if not args.amp:
        return nullcontext()
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type=args.device.type, dtype=dtype)


def _reduce_sums(values: list[float], device: torch.device) -> list[float]:
    tensor = torch.tensor(values, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.tolist()


def train_one_epoch(
    args,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineLRScheduler,
    criterion: nn.Module,
    mixup_fn: Optional[Mixup],
    model_ema: ModelEmaV3,
    loss_scaler: Optional[NativeScaler],
    updates_per_epoch: int,
    num_updates: int,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = correct1 = correct5 = samples = 0.0
    micro_batches = updates_per_epoch * args.grad_accum_steps
    consumed = 0
    for batch_index, (images, hard_targets) in enumerate(loader):
        if batch_index >= micro_batches:
            break
        images = images.to(args.device, non_blocking=True)
        hard_targets = hard_targets.to(args.device, non_blocking=True)
        targets = hard_targets
        if mixup_fn is not None:
            images, targets = mixup_fn(images, targets)

        should_update = (batch_index + 1) % args.grad_accum_steps == 0
        sync_context = (
            model.no_sync()
            if isinstance(model, DDP) and not should_update
            else nullcontext()
        )
        with sync_context:
            with _autocast(args):
                logits, _ = model(images)
                loss = criterion(logits, targets)
                backward_loss = loss / args.grad_accum_steps
            if loss_scaler is not None:
                loss_scaler(
                    backward_loss,
                    optimizer,
                    parameters=model.parameters(),
                    need_update=should_update,
                )
            else:
                backward_loss.backward()
                if should_update:
                    optimizer.step()

        batch_size = hard_targets.numel()
        predictions = logits.detach()
        loss_sum += loss.detach().item() * batch_size
        correct1 += predictions.argmax(1).eq(hard_targets).sum().item()
        correct5 += (
            predictions.topk(min(5, predictions.shape[1]), dim=1)
            .indices.eq(hard_targets[:, None]).any(1).sum().item()
        )
        samples += batch_size
        consumed += 1

        if should_update:
            optimizer.zero_grad(set_to_none=True)
            num_updates += 1
            model_ema.update(_unwrap(model), step=num_updates)
            scheduler.step_update(num_updates)

    if consumed != micro_batches:
        raise RuntimeError(
            f"Train loader produced {consumed} micro-batches, expected {micro_batches}"
        )
    loss_sum, correct1, correct5, samples = _reduce_sums(
        [loss_sum, correct1, correct5, samples], args.device
    )
    return {
        "loss": loss_sum / samples,
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
            logits, _ = model(images)
            loss = criterion(logits, targets)
        batch_size = targets.numel()
        loss_sum += loss.item() * batch_size
        correct1 += logits.argmax(1).eq(targets).sum().item()
        correct5 += (
            logits.topk(min(5, logits.shape[1]), dim=1)
            .indices.eq(targets[:, None]).any(1).sum().item()
        )
        samples += batch_size
    loss_sum, correct1, correct5, samples = _reduce_sums(
        [loss_sum, correct1, correct5, samples], args.device
    )
    return {
        "loss": loss_sum / samples,
        "acc1": 100.0 * correct1 / samples,
        "acc5": 100.0 * correct5 / samples,
    }


def capture_rng_state(args) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
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


def restore_rng_state(args, checkpoint: dict[str, Any]) -> None:
    states = checkpoint.get("rng_by_rank")
    if not isinstance(states, list) or len(states) != args.world_size:
        raise ValueError("Resume checkpoint RNG state does not match the current world size")
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


def init_wandb(args, config: dict[str, Any], checkpoint: Optional[dict[str, Any]]):
    if not is_primary(args) or args.wandb_mode == "disabled":
        return None
    saved = checkpoint.get("wandb", {}) if checkpoint is not None else {}
    run_id = saved.get("run_id") or args.wandb_run_id or None
    wandb_dir = Path(args.wandb_dir)
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(
        project=saved.get("project") or args.wandb_project,
        entity=saved.get("entity") or args.wandb_entity or None,
        name=saved.get("name") or args.wandb_name or None,
        group=saved.get("group") or args.wandb_group or None,
        id=run_id,
        resume="allow" if run_id else None,
        mode=args.wandb_mode,
        dir=str(wandb_dir),
        config=None if checkpoint is not None and run_id else config,
    )


def wandb_metadata(run) -> dict[str, Any]:
    if run is None:
        return {}
    return {
        "run_id": run.id,
        "project": run.project,
        "entity": run.entity,
        "name": run.name,
        "group": run.group,
    }


def build_config(
    args,
    model: RecurrentCNN,
    num_classes: int,
    train_size: int,
    val_size: int,
    updates_per_epoch: int,
) -> dict[str, Any]:
    mismatches = official_recipe_mismatches(args)
    paper_model_exact = (
        args.convnext_version == 1
        and model.stage_depths == NATIVE_TINY_DEPTHS
        and model.stage_repeats == NATIVE_TINY_REPEATS
        and not any(model.reg_mode)
    )
    return {
        "model_family": MODEL_FAMILY,
        "model_arch": model_arch_name(
            args.convnext_version,
            model.reg_mode,
            delta_mode=model.delta_mode,
            reg_head=model.reg_head,
        ),
        "recipe": RECIPE_NAME,
        "training_recipe_exact": not mismatches and not args.smoke,
        "training_recipe_mismatches": mismatches,
        "paper_model_exact": paper_model_exact,
        "architecture": {
            "convnext_version": args.convnext_version,
            "arr1": list(model.stage_depths),
            "arr2": list(model.stage_repeats),
            "reg_mode": list(model.reg_mode),
            "n_reg": list(model.n_reg),
            "delta_mode": model.delta_mode,
            "reg_head": model.reg_head,
            "delta_backend": DELTA_BACKEND if model.delta_mode else None,
            "delta_chunk_size": DELTA_CHUNK_SIZE if model.delta_mode else None,
            "delta_conv_size": DELTA_CONV_SIZE if model.delta_mode else None,
            "delta_norm_eps": DELTA_NORM_EPS if model.delta_mode else None,
            "register_readout": (
                "concat_mean_reg_feature" if model.reg_head else "feature_mean"
            ),
            "unique_blocks": model.unique_blocks,
            "block_applications": model.block_applications,
            "register_stage_count": model.register_stage_count,
            "register_applications": model.register_applications,
            "register_attention": "rats_shared_qkv_identity_registers",
            "register_heads": REG_HEADS,
            "register_sdpa_backend": REG_SDPA_BACKEND,
            "register_mlp_ratio": 4,
            "register_data_term": False,
            "register_reconstruction": False,
            "register_layerscale": False,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "drop_path_rate": model.drop_path_rate,
            "drop_path_schedule": model.drop_path_schedule,
            "drop_path_rates": list(model.drop_path_rates),
        },
        "training": {
            **_plain_args(args),
            "optimizer": "adamw",
            "optimizer_betas": [0.9, 0.999],
            "optimizer_eps": 1e-8,
            "peak_lr": scaled_peak_lr(args),
            "effective_batch_size": effective_batch_size(args),
            "updates_per_epoch": updates_per_epoch,
            "total_updates": args.epochs * updates_per_epoch,
            "gradient_clip": None,
            "layerwise_lr_decay": None,
            "mixup_prob": 1.0,
            "mixup_switch_prob": 0.5,
            "mixup_mode": "batch",
            "random_erasing_mode": "pixel",
            "random_erasing_count": 1,
            "train_interpolation": "bicubic",
            "validation_crop_pct": 0.875,
        },
        "dataset": {
            "num_classes": num_classes,
            "train_size": train_size,
            "val_size": val_size,
        },
    }


def checkpoint_state(
    args,
    config: dict[str, Any],
    model: nn.Module,
    model_ema: ModelEmaV3,
    optimizer,
    scheduler,
    loss_scaler,
    epoch: int,
    num_updates: int,
    best_ema_acc1: float,
    rng_by_rank,
    run,
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_family": MODEL_FAMILY,
        "epoch": epoch,
        "num_updates": num_updates,
        "best_ema_acc1": best_ema_acc1,
        "world_size": args.world_size,
        "arguments": _plain_args(args),
        "config": config,
        "model": _unwrap(model).state_dict(),
        "model_ema": model_ema.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": loss_scaler.state_dict() if loss_scaler is not None else None,
        "rng_by_rank": rng_by_rank,
        "wandb": wandb_metadata(run),
    }


def restore_training_state(checkpoint, model, model_ema, optimizer, scheduler, loss_scaler):
    model.load_state_dict(checkpoint["model"], strict=True)
    model_ema.module.load_state_dict(checkpoint["model_ema"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if loss_scaler is not None and checkpoint.get("scaler") is not None:
        loss_scaler.load_state_dict(checkpoint["scaler"])
    start_epoch = int(checkpoint["epoch"]) + 1
    num_updates = int(checkpoint["num_updates"])
    scheduler.step_update(num_updates)
    return start_epoch, num_updates, float(checkpoint.get("best_ema_acc1", -1.0))


def synchronized_stop_requested(args) -> bool:
    requested = torch.tensor(
        int(_STOP_REQUESTED), device=args.device, dtype=torch.int32
    )
    if dist.is_initialized():
        dist.all_reduce(requested, op=dist.ReduceOp.MAX)
    return bool(requested.item())


def default_run_name(args) -> str:
    return f"convnext-official-ep{args.epochs}-warmup{args.warmup_epochs}"


def architecture_suffix(
    depths,
    repeats,
    reg_mode=DEFAULT_REG_MODE,
    n_reg=DEFAULT_N_REG,
    delta_mode=False,
    reg_head=False,
) -> str:
    arr1 = "-".join(map(str, depths))
    arr2 = "-".join(map(str, repeats))
    modes, counts = validate_register_arrays(reg_mode, n_reg, depths)
    suffix = f"ARR1-{arr1}_ARR2-{arr2}"
    if any(modes):
        reg = "-".join(map(str, modes))
        nreg = "-".join(map(str, counts))
        suffix += f"_REG-{reg}_NREG-{nreg}"
    if delta_mode:
        suffix += "_DELTA1"
    if reg_head:
        suffix += "_REGHEAD1"
    return suffix


def append_architecture_suffix(
    value: str | Path,
    depths,
    repeats,
    reg_mode=DEFAULT_REG_MODE,
    n_reg=DEFAULT_N_REG,
    delta_mode=False,
    reg_head=False,
) -> str:
    value = str(value)
    suffix = architecture_suffix(
        depths,
        repeats,
        reg_mode,
        n_reg,
        delta_mode=delta_mode,
        reg_head=reg_head,
    )
    return value if suffix in value else f"{value}_{suffix}"


def default_output_dir(args, depths, repeats) -> str:
    return (
        f"outputs/imagenet_recurrent_official_convnextV{args.convnext_version}_"
        f"{architecture_suffix(depths, repeats, args.reg_mode, args.n_reg, args.delta_mode, args.reg_head)}_"
        f"ep{args.epochs}_warmup{args.warmup_epochs}_"
        f"gbs{effective_batch_size(args)}_seed{args.seed}"
    )


def run(args) -> None:
    setup_logging_once()
    install_signal_handlers()
    args.device = init_distributed_device(args)
    if args.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if args.resume and args.smoke:
        raise ValueError("--resume and --smoke are mutually exclusive")
    resume_checkpoint = _load_checkpoint(args.resume) if args.resume else None
    if resume_checkpoint is not None:
        if int(resume_checkpoint.get("world_size", -1)) != args.world_size:
            raise ValueError(
                f"Resume world size mismatch: checkpoint={resume_checkpoint.get('world_size')} "
                f"current={args.world_size}"
            )
        _restore_resume_arguments(args, resume_checkpoint)
    if args.smoke:
        _apply_smoke_defaults(args)
    if args.device.type != "cuda" and args.amp:
        LOG.warning("Disabling AMP on non-CUDA device %s", args.device)
        args.amp = False
    if resume_checkpoint is not None:
        saved_amp = bool(resume_checkpoint["arguments"]["amp"])
        if args.amp != saved_amp:
            raise ValueError(
                "Resume AMP mode cannot be preserved on the selected device: "
                f"checkpoint={saved_amp}, effective={args.amp}"
            )

    depths, repeats = validate_stage_arrays(args.arr1, args.arr2)
    reg_mode, n_reg = validate_register_arrays(
        args.reg_mode,
        args.n_reg,
        depths,
    )
    args.reg_mode = ",".join(map(str, reg_mode))
    args.n_reg = ",".join(map(str, n_reg))
    if not args.resume:
        output_base = args.output_dir or default_output_dir(args, depths, repeats)
        args.output_dir = append_architecture_suffix(
            output_base,
            depths,
            repeats,
            reg_mode,
            n_reg,
            delta_mode=args.delta_mode,
            reg_head=args.reg_head,
        )
        args.wandb_project = append_architecture_suffix(
            args.wandb_project,
            depths,
            repeats,
            reg_mode,
            n_reg,
            delta_mode=args.delta_mode,
            reg_head=args.reg_head,
        )
    args.wandb_name = args.wandb_name or default_run_name(args)
    if args.batch_size < 1 or args.grad_accum_steps < 1:
        raise ValueError("batch_size and grad_accum_steps must be positive")
    if args.reference_batch_size < 1:
        raise ValueError("reference_batch_size must be positive")
    if args.epochs < 1 or not 0 <= args.warmup_epochs <= args.epochs:
        raise ValueError("epochs must be positive and warmup_epochs must be in [0, epochs]")
    if min(args.base_lr, args.warmup_lr, args.min_lr, args.weight_decay) < 0:
        raise ValueError("learning rates and weight_decay must be non-negative")
    if args.mixup < 0 or args.cutmix < 0:
        raise ValueError("mixup and cutmix must be non-negative")
    if not 0 <= args.smoothing < 1 or not 0 <= args.reprob <= 1:
        raise ValueError("smoothing must be in [0, 1) and reprob in [0, 1]")
    if not 0 <= args.ema_decay < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    if args.save_every < 0:
        raise ValueError("save_every must be non-negative")
    mismatches = official_recipe_mismatches(args)
    if args.strict_official_recipe and mismatches and not args.smoke:
        raise ValueError(
            "Strict official recipe check failed:\n  " + "\n  ".join(mismatches)
        )

    random_seed(args.seed, args.rank)
    train_loader, val_loader, train_sampler, num_classes, train_size, val_size = create_loaders(args)
    if num_classes != 1000 and args.strict_official_recipe and not args.smoke:
        raise ValueError(f"Expected 1000 ImageNet classes, found {num_classes}")
    updates_per_epoch = train_size // effective_batch_size(args)
    if updates_per_epoch < 1:
        raise ValueError(
            f"Training set size {train_size} is smaller than effective batch "
            f"{effective_batch_size(args)}"
        )
    if len(train_loader) < updates_per_epoch * args.grad_accum_steps:
        raise ValueError("Train loader cannot provide the requested full optimizer updates")

    model = RecurrentCNN(
        depths,
        repeats,
        num_classes=num_classes,
        convnext_version=args.convnext_version,
        drop_path_rate=args.drop_path_rate,
        reg_mode=reg_mode,
        n_reg=n_reg,
        delta_mode=args.delta_mode,
        reg_head=args.reg_head,
    ).to(args.device)
    config = build_config(args, model, num_classes, train_size, val_size, updates_per_epoch)

    if args.resume:
        saved_config = resume_checkpoint.get("config")
        if not isinstance(saved_config, dict):
            raise ValueError("Resume checkpoint has no valid config")
        saved_config = dict(saved_config)
        saved_architecture = dict(saved_config.get("architecture", {}))
        saved_architecture.setdefault("reg_mode", list(DEFAULT_REG_MODE))
        saved_architecture.setdefault("n_reg", list(DEFAULT_N_REG))
        saved_architecture.setdefault("delta_mode", False)
        saved_architecture.setdefault("reg_head", False)
        saved_architecture.setdefault("delta_backend", None)
        saved_architecture.setdefault("delta_chunk_size", None)
        saved_architecture.setdefault("delta_conv_size", None)
        saved_architecture.setdefault("delta_norm_eps", None)
        saved_architecture.setdefault("register_readout", "feature_mean")
        saved_architecture.setdefault("register_stage_count", 0)
        saved_architecture.setdefault("register_applications", 0)
        saved_architecture.setdefault(
            "register_attention", "rats_shared_qkv_identity_registers"
        )
        saved_architecture.setdefault("register_heads", REG_HEADS)
        saved_architecture.setdefault("register_sdpa_backend", REG_SDPA_BACKEND)
        saved_architecture.setdefault("register_mlp_ratio", 4)
        saved_architecture.setdefault("register_data_term", False)
        saved_architecture.setdefault("register_reconstruction", False)
        saved_architecture.setdefault("register_layerscale", False)
        saved_config["architecture"] = saved_architecture
        for key in ("model_arch", "recipe", "architecture", "dataset"):
            if saved_config.get(key) != config.get(key):
                raise ValueError(f"Resume {key} differs from the checkpoint")
        saved_training = saved_config.get("training", {})
        for key in ("effective_batch_size", "updates_per_epoch", "peak_lr"):
            if saved_training.get(key) != config["training"].get(key):
                raise ValueError(f"Resume training.{key} differs from the checkpoint")
    output_dir = Path(args.output_dir)
    latest_path = output_dir / "checkpoint_latest.pt"
    if not args.resume and latest_path.exists():
        raise FileExistsError(
            f"{latest_path} already exists; use --resume or a new output directory"
        )
    if is_primary(args):
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json_save(config, output_dir / "config.json")
    if dist.is_initialized():
        dist.barrier()

    peak_lr = scaled_peak_lr(args)
    optimizer = create_optimizer(args, model)
    scheduler = create_update_scheduler(args, optimizer, updates_per_epoch)
    model_ema = ModelEmaV3(
        model,
        decay=args.ema_decay,
        use_warmup=False,
        foreach=args.device.type == "cuda",
    )
    loss_scaler = (
        NativeScaler(device=args.device.type)
        if args.amp and args.amp_dtype == "float16" and args.device.type == "cuda"
        else None
    )
    start_epoch, num_updates, best_ema_acc1 = 0, 0, -1.0
    if resume_checkpoint is not None:
        start_epoch, num_updates, best_ema_acc1 = restore_training_state(
            resume_checkpoint,
            model,
            model_ema,
            optimizer,
            scheduler,
            loss_scaler,
        )

    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.device.index] if args.device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    mixup_fn = create_mixup(args, num_classes)
    criterion = create_train_criterion(args)
    run_handle = init_wandb(args, config, resume_checkpoint)
    metrics_path = output_dir / "metrics.jsonl"
    if is_primary(args):
        prepare_metrics_file(metrics_path, start_epoch, bool(args.resume))

    if is_primary(args):
        LOG.info(
            "model=convnextV%d ARR1=%s ARR2=%s REG_MODE=%s N_REG=%s "
            "DELTA_MODE=%s REG_HEAD=%s "
            "params=%d paper_model_exact=%s",
            args.convnext_version,
            depths,
            repeats,
            reg_mode,
            n_reg,
            args.delta_mode,
            args.reg_head,
            config["architecture"]["parameters"],
            config["paper_model_exact"],
        )
        LOG.info(
            "effective_batch_size=%d peak_lr=%g updates_per_epoch=%d "
            "training_recipe_exact=%s output=%s",
            effective_batch_size(args),
            peak_lr,
            updates_per_epoch,
            config["training_recipe_exact"],
            output_dir,
        )

    if resume_checkpoint is not None:
        restore_rng_state(args, resume_checkpoint)

    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_metrics, num_updates = train_one_epoch(
            args,
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            mixup_fn,
            model_ema,
            loss_scaler,
            updates_per_epoch,
            num_updates,
        )
        raw_metrics = evaluate(args, model, val_loader)
        ema_metrics = evaluate(args, model_ema.module, val_loader)
        improved = ema_metrics["acc1"] > best_ema_acc1
        best_ema_acc1 = max(best_ema_acc1, ema_metrics["acc1"])
        elapsed = time.time() - epoch_started
        record = {
            "epoch": epoch,
            "num_updates": num_updates,
            "lr": epoch_lr,
            "train": train_metrics,
            "raw": raw_metrics,
            "ema": ema_metrics,
            "best_ema_acc1": best_ema_acc1,
            "epoch_sec": elapsed,
        }
        if is_primary(args):
            append_metric(metrics_path, record)
            if run_handle is not None:
                run_handle.log(record, step=epoch + 1)
        rng_by_rank = collect_rng_states(args)
        if is_primary(args):
            state = checkpoint_state(
                args,
                config,
                model,
                model_ema,
                optimizer,
                scheduler,
                loss_scaler,
                epoch,
                num_updates,
                best_ema_acc1,
                rng_by_rank,
                run_handle,
            )
            atomic_torch_save(state, latest_path)
            if improved:
                duplicate_checkpoint(latest_path, output_dir / "checkpoint_best.pt")
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                duplicate_checkpoint(
                    latest_path,
                    output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt",
                )
            if epoch + 1 == args.epochs:
                duplicate_checkpoint(latest_path, output_dir / "checkpoint_final.pt")
            LOG.info(
                "epoch=%d/%d train_loss=%.4f raw_acc1=%.3f ema_acc1=%.3f "
                "best_ema=%.3f lr=%.3e sec=%.1f",
                epoch + 1,
                args.epochs,
                train_metrics["loss"],
                raw_metrics["acc1"],
                ema_metrics["acc1"],
                best_ema_acc1,
                epoch_lr,
                elapsed,
            )
        if synchronized_stop_requested(args):
            if is_primary(args):
                LOG.warning("Stopping cleanly after epoch %d", epoch + 1)
            break

    if start_epoch >= args.epochs and is_primary(args):
        LOG.info("Checkpoint already completed all %d epochs", args.epochs)
    if run_handle is not None:
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
