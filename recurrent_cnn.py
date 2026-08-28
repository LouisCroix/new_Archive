"""Tied recurrent ResNet/ConvNeXt baselines for ImageNet-1K.

Naive mode repeats native first-stage blocks. Pro mode uses native early stages
as a feed-forward frontend, then repeats third-stage blocks at 14x14 resolution.
The complete BLOCK_DEPTH-block recurrent cell is shared across all T iterations.
ResNet BatchNorm affine parameters remain tied, while recurrent running
statistics are iteration-specific because recurrent states are not identically
distributed.
"""

from contextlib import nullcontext
from functools import partial
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models.convnext import CNBlock, LayerNorm2d
from torchvision.models.resnet import BasicBlock

from imagenet_data import make_imagenet_loaders

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


ARCHITECTURES = {
    "resnet": {
        "naive": {
            "model_arch": "recurrent_resnet18_first_basicblock_tied_v2",
            "width": 64,
            "stem": "resnet18_conv7_s2_bn_relu_maxpool_s2",
            "frontend": "identity",
            "block": "torchvision_resnet18_basicblock_stage1_block1_iteration_bn",
            "feature_resolution": 56,
            "normalization_state": "per_iteration_running_stats_shared_affine",
        },
        "pro": {
            "model_arch": "recurrent_resnet18_stage3_basicblock_tied_v3",
            "width": 256,
            "stem": "resnet18_conv7_s2_bn_relu_maxpool_s2",
            "frontend": "resnet18_through_layer3_block1",
            "block": "torchvision_resnet18_stage3_basicblock_iteration_bn",
            "feature_resolution": 14,
            "normalization_state": "per_iteration_running_stats_shared_affine",
        },
    },
    "convnext": {
        "naive": {
            "model_arch": "recurrent_convnext_tiny_first_cnblock_tied_v2",
            "width": 96,
            "stem": "convnext_tiny_conv4_s4_layernorm2d",
            "frontend": "identity",
            "block": "torchvision_convnext_tiny_stage1_block1",
            "feature_resolution": 56,
            "normalization_state": "stateless_layernorm_per_call",
        },
        "pro": {
            "model_arch": "recurrent_convnext_tiny_stage3_cnblock_tied_v3",
            "width": 384,
            "stem": "convnext_tiny_conv4_s4_layernorm2d",
            "frontend": "convnext_tiny_stages1_2_and_downsample_to_stage3",
            "block": "torchvision_convnext_tiny_stage3_cnblock",
            "feature_resolution": 14,
            "normalization_state": "stateless_layernorm_per_call",
        },
    },
}


def env_flag(name, default=0):
    return bool(int(os.environ.get(name, default)))


class IterationBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm with shared affine parameters and one buffer bank per iteration.

    A tied recurrent block sees a different activation distribution at each
    iteration. Sharing gamma/beta preserves weight tying, while separate running
    statistics make training-mode and evaluation-mode normalization consistent.
    """

    def __init__(self, num_features, iterations, **kwargs):
        if iterations < 1:
            raise ValueError(f"iterations={iterations} must be positive")
        super().__init__(num_features, **kwargs)
        self.iterations = iterations
        self.iteration = 0
        if self.track_running_stats:
            factory_kwargs = {"device": self.running_mean.device, "dtype": self.running_mean.dtype}
            self.running_mean = torch.zeros(iterations, num_features, **factory_kwargs)
            self.running_var = torch.ones(iterations, num_features, **factory_kwargs)
            self.num_batches_tracked = torch.zeros(
                iterations, dtype=torch.long, device=self.running_mean.device
            )

    def set_iteration(self, iteration):
        if not 0 <= iteration < self.iterations:
            raise IndexError(
                f"BatchNorm iteration={iteration} is outside [0, {self.iterations})"
            )
        self.iteration = iteration

    def forward(self, inputs):
        self._check_input_dim(inputs)
        exponential_average_factor = 0.0 if self.momentum is None else self.momentum
        running_mean = running_var = None
        if self.track_running_stats:
            running_mean = self.running_mean[self.iteration]
            running_var = self.running_var[self.iteration]
            if self.training:
                self.num_batches_tracked[self.iteration].add_(1)
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked[self.iteration]
                    )
        use_batch_stats = self.training or not self.track_running_stats
        return F.batch_norm(
            inputs,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            use_batch_stats,
            exponential_average_factor,
            self.eps,
        )


def adamw_parameter_groups(model, weight_decay):
    """Match ConvNeXt's native AdamW filtering of 1-D parameters and biases."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = no_decay if parameter.ndim == 1 or name.endswith(".bias") else decay
        target.append(parameter)
    if not decay or not no_decay:
        raise ValueError("Expected both decay and no-decay AdamW parameter groups")
    return [
        {"params": decay, "weight_decay": weight_decay, "group_name": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"},
    ]


class RecurrentCNN(nn.Module):
    """A native frontend followed by a weight-tied native residual cell."""

    def __init__(
        self,
        block_type="resnet",
        iterations=12,
        num_classes=1000,
        mode="naive",
        block_depth=1,
    ):
        super().__init__()
        if block_type not in ARCHITECTURES:
            raise ValueError(
                f"Unsupported BLOCK_TYPE={block_type!r}; use resnet or convnext"
            )
        if iterations < 1:
            raise ValueError(f"T={iterations} must be positive")
        if mode not in {"naive", "pro"}:
            raise ValueError(f"Unsupported mode={mode!r}; use naive or pro")
        if block_depth < 1:
            raise ValueError(f"BLOCK_DEPTH={block_depth} must be positive")
        self.block_type = block_type
        self.iterations = iterations
        self.mode = mode
        self.block_depth = block_depth
        self.architecture = ARCHITECTURES[block_type][mode]

        if block_type == "resnet":
            stem_width = 64
            width = self.architecture["width"]
            recurrent_norm = partial(IterationBatchNorm2d, iterations=iterations)
            self.stem = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.frontend = (
                nn.Identity() if mode == "naive" else self._make_resnet_pro_frontend()
            )
            blocks = [
                BasicBlock(width, width, norm_layer=recurrent_norm)
                for _ in range(block_depth)
            ]
            # Keep the depth-1 state_dict compatible with completed v2 runs.
            self.block = blocks[0] if block_depth == 1 else nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head_norm = nn.Identity()
            self.head = nn.Linear(width, num_classes)
            self.apply(self._init_resnet)
        else:
            stem_width = 96
            width = self.architecture["width"]
            stem_norm = partial(LayerNorm2d, eps=1e-6)
            self.stem = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=4, stride=4, bias=True),
                stem_norm(stem_width),
            )
            self.frontend = (
                nn.Identity() if mode == "naive" else self._make_convnext_pro_frontend()
            )
            blocks = [self._make_convnext_block(width) for _ in range(block_depth)]
            # Keep the depth-1 state_dict compatible with completed v2 runs.
            self.block = blocks[0] if block_depth == 1 else nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.head_norm = nn.LayerNorm(width, eps=1e-6)
            self.head = nn.Linear(width, num_classes)
            self.apply(self._init_convnext)

    @staticmethod
    def _resnet_block(inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        return BasicBlock(
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            norm_layer=nn.BatchNorm2d,
        )

    @classmethod
    def _make_resnet_pro_frontend(cls):
        return nn.Sequential(
            cls._resnet_block(64, 64),
            cls._resnet_block(64, 64),
            cls._resnet_block(64, 128, stride=2),
            cls._resnet_block(128, 128),
            cls._resnet_block(128, 256, stride=2),
        )

    @staticmethod
    def _make_convnext_block(width):
        return CNBlock(
            width,
            layer_scale=1e-6,
            stochastic_depth_prob=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

    @classmethod
    def _make_convnext_pro_frontend(cls):
        norm = partial(LayerNorm2d, eps=1e-6)
        return nn.Sequential(
            nn.Sequential(*(cls._make_convnext_block(96) for _ in range(3))),
            nn.Sequential(norm(96), nn.Conv2d(96, 192, kernel_size=2, stride=2)),
            nn.Sequential(*(cls._make_convnext_block(192) for _ in range(3))),
            nn.Sequential(norm(192), nn.Conv2d(192, 384, kernel_size=2, stride=2)),
        )

    @staticmethod
    def _init_resnet(module):
        # Matches torchvision.models.resnet.ResNet.__init__.
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    @staticmethod
    def _init_convnext(module):
        # Matches torchvision.models.convnext.ConvNeXt.__init__.
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, log_residuals=False):
        x = self.stem(x)
        x = self.frontend(x)
        residuals = []
        for iteration in range(self.iterations):
            if self.block_type == "resnet":
                for module in self.block.modules():
                    if isinstance(module, IterationBatchNorm2d):
                        module.set_iteration(iteration)
            previous = x
            x = self.block(x)
            if log_residuals:
                numerator = (x - previous).float().flatten(1).norm(dim=1).mean()
                denominator = x.float().flatten(1).norm(dim=1).mean().clamp_min(1e-8)
                residuals.append((numerator / denominator).item())
        x = self.pool(x).flatten(1)
        return self.head(self.head_norm(x)), residuals


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if env_flag("REQUIRE_CUDA", 0) and not torch.cuda.is_available():
        raise RuntimeError(
            "REQUIRE_CUDA=1 but torch.cuda.is_available() is False; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"cuda_device_count={torch.cuda.device_count()}"
        )
    distributed = world_size > 1
    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return distributed, rank, local_rank, world_size


def load_resume_config(path):
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict) or config.get("model_family") != "recurrent_cnn":
        raise ValueError(f"Not a recurrent_cnn checkpoint: {path}")
    checkpoint_version = config.get("experiment_version")
    if checkpoint_version not in {2, 3}:
        if config.get("block_type") == "resnet":
            incompatibility = (
                "its tied ResNet block has only one mixed BatchNorm "
                "running-statistics bank and its optimizer uses the old parameter groups"
            )
        else:
            incompatibility = "its optimizer uses the old unfiltered parameter groups"
        raise ValueError(
            "This checkpoint predates the recurrent normalization fix and cannot be "
            f"resumed exactly: {incompatibility}. Start a new run instead."
        )
    if checkpoint_version == 2:
        # v2 had only the architecture now named naive with one recurrent block.
        os.environ["MODE"] = "naive"
        os.environ["BLOCK_DEPTH"] = "1"
    env_from_config = {
        "BLOCK_TYPE": "block_type",
        "MODE": "mode",
        "BLOCK_DEPTH": "block_depth",
        "T": "T",
        "DATA_ROOT": "data_root",
        "IMG": "img",
        "RESIZE": "resize",
        "EPOCHS": "epochs",
        "BS": "batch_size",
        "WORKERS": "workers",
        "LIMIT_TRAIN": "limit_train",
        "LIMIT_VAL": "limit_val",
        "MAX_LR": "max_lr",
        "MIN_LR": "min_lr",
        "WARMUP_EPOCHS": "warmup_epochs",
        "GRAD_ACCUM_STEPS": "grad_accum_steps",
        "AMP": "amp",
        "AMP_DTYPE": "amp_dtype",
    }
    for env_name, config_name in env_from_config.items():
        value = config.get(config_name)
        if value is not None:
            os.environ[env_name] = str(int(value) if isinstance(value, bool) else value)
    return checkpoint


def main():
    distributed, rank, local_rank, world_size = setup_distributed()

    def is_main():
        return rank == 0

    def log(*args, **kwargs):
        if is_main():
            print(*args, **kwargs)

    resume_path = os.environ.get("RESUME", "").strip()
    resume_checkpoint = load_resume_config(resume_path)
    block_type = os.environ.get("BLOCK_TYPE", "resnet").lower()
    if block_type not in ARCHITECTURES:
        raise ValueError(f"Unsupported BLOCK_TYPE={block_type}; use resnet or convnext")
    mode = os.environ.get("MODE", "naive").lower()
    if mode not in {"naive", "pro"}:
        raise ValueError(f"Unsupported MODE={mode}; use naive or pro")
    block_depth = int(os.environ.get("BLOCK_DEPTH", 1))
    if block_depth < 1:
        raise ValueError(f"BLOCK_DEPTH={block_depth} must be positive")
    arch = ARCHITECTURES[block_type][mode]
    iterations = int(os.environ.get("T", 12))
    if iterations < 1:
        raise ValueError(f"T={iterations} must be positive")

    if env_flag("PRINT_MODEL_INFO", 0):
        model = RecurrentCNN(block_type, iterations, 1000, mode, block_depth)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(json.dumps({
            "experiment_version": 3,
            "block_type": block_type,
            "mode": mode,
            "block_depth": block_depth,
            "iterations": iterations,
            "parameters": total,
            "trainable_parameters": trainable,
            **arch,
        }, indent=2))
        if distributed:
            dist.destroy_process_group()
        return

    device = (
        f"cuda:{local_rank}" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    device_type = "cuda" if str(device).startswith("cuda") else ("mps" if device == "mps" else "cpu")
    amp_enabled = env_flag("AMP", 0) and device_type in {"cuda", "cpu"}
    amp_dtype_name = os.environ.get("AMP_DTYPE", "bfloat16").lower()
    amp_dtypes = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    amp_dtype = amp_dtypes.get(amp_dtype_name)
    if amp_enabled and amp_dtype is None:
        raise ValueError(f"Unsupported AMP_DTYPE={amp_dtype_name}; use bfloat16 or float16")
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled and amp_dtype == torch.float16 and device_type == "cuda",
    )

    def autocast_ctx():
        if amp_enabled:
            return torch.autocast(device_type=device_type, dtype=amp_dtype)
        return nullcontext()

    data_root = os.environ.get("DATA_ROOT", "/cis/project/peq_project/imagenet-1k")
    img = int(os.environ.get("IMG", 224))
    resize = int(os.environ.get("RESIZE", 256))
    epochs = int(os.environ.get("EPOCHS", 22))
    batch_size = int(os.environ.get("BS", 128))
    workers = int(os.environ.get("WORKERS", 4))
    dataloader_timeout = int(os.environ.get("DATALOADER_TIMEOUT", 0))
    max_lr = float(os.environ.get("MAX_LR", 5e-4))
    min_lr = float(os.environ.get("MIN_LR", 1e-6))
    warmup_epochs = int(os.environ.get("WARMUP_EPOCHS", 2))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 2))
    limit_train = int(os.environ.get("LIMIT_TRAIN", 0))
    limit_val = int(os.environ.get("LIMIT_VAL", 0))
    seeds = [int(value) for value in os.environ.get("SEEDS", "0").split(",")]
    progress = env_flag("PROGRESS", 1)
    memory_probe = env_flag("MEMORY_PROBE", 0)
    if warmup_epochs < 0:
        raise ValueError("WARMUP_EPOCHS must be non-negative")
    if grad_accum_steps < 1:
        raise ValueError("GRAD_ACCUM_STEPS must be positive")
    if dataloader_timeout < 0:
        raise ValueError("DATALOADER_TIMEOUT must be non-negative")
    if env_flag("SMOKE", 0):
        if resume_path:
            raise ValueError("SMOKE cannot be combined with RESUME")
        epochs, seeds = 1, [0]
        limit_train = limit_train or 2048
        limit_val = limit_val or 1024

    output_dir = os.environ.get(
        "OUTPUT_DIR",
        f"outputs/imagenet_recurrent_v3_{mode}_{block_type}_depth{block_depth}_"
        f"T{iterations}_img{img}_epochs{epochs}_"
        f"BS{batch_size}_accum{grad_accum_steps}_lr{max_lr}_minlr{min_lr}",
    )
    if resume_checkpoint is not None:
        saved_seed = int(resume_checkpoint.get("seed", resume_checkpoint["config"].get("seed", 0)))
        seeds = [saved_seed]
        output_dir = os.path.dirname(os.path.dirname(os.path.abspath(resume_path)))

    train_loader, val_loader, num_classes, train_sampler = make_imagenet_loaders(
        data_root,
        img,
        resize,
        batch_size,
        workers,
        limit_train,
        limit_val,
        distributed,
        rank,
        world_size,
        persistent_workers=False,
        timeout=dataloader_timeout,
    )

    def metric_sums(values):
        stats = torch.tensor(values, device=device, dtype=torch.float64)
        if distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats.tolist()

    def unwrap(model):
        return model.module if isinstance(model, DDP) else model

    def config(seed, params):
        return {
            "model_family": "recurrent_cnn",
            "experiment_version": 3,
            "model_arch": arch["model_arch"],
            "block_type": block_type,
            "mode": mode,
            "block_depth": block_depth,
            "stem": arch["stem"],
            "frontend": arch["frontend"],
            "block": arch["block"],
            "normalization_state": arch["normalization_state"],
            "width": arch["width"],
            "feature_resolution": arch["feature_resolution"],
            "weight_tying": (
                "one_shared_block_across_all_iterations"
                if block_depth == 1
                else "one_shared_block_cell_across_all_iterations"
            ),
            "T": iterations,
            "activation_checkpointing": False,
            "data_root": data_root,
            "img": img,
            "resize": resize,
            "epochs": epochs,
            "batch_size": batch_size,
            "workers": workers,
            "limit_train": limit_train,
            "limit_val": limit_val,
            "max_lr": max_lr,
            "min_lr": min_lr,
            "warmup_epochs": warmup_epochs,
            "grad_accum_steps": grad_accum_steps,
            "optimizer": "adamw",
            "weight_decay": 0.05,
            "weight_decay_filter": "no_decay_for_1d_parameters_and_bias",
            "label_smoothing": 0.1,
            "amp": amp_enabled,
            "amp_dtype": amp_dtype_name,
            "num_classes": num_classes,
            "seed": seed,
            "parameters": params,
            "params_m": params / 1e6,
        }

    def save_json(path, value):
        if not is_main():
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = f"{path}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)
        os.replace(temporary, path)

    def save_curves(path, records):
        if not is_main() or not records:
            return
        try:
            os.makedirs(os.path.join(output_dir, ".matplotlib"), exist_ok=True)
            os.environ.setdefault("MPLCONFIGDIR", os.path.join(output_dir, ".matplotlib"))
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            log(f"warning: could not save curves: {exc}")
            return
        xs = [record["epoch"] for record in records]
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].plot(xs, [r["train_loss"] for r in records], label="train")
        axes[0, 0].plot(xs, [r["val_loss"] for r in records], label="val")
        axes[0, 1].plot(xs, [r["train_acc1"] for r in records], label="train@1")
        axes[0, 1].plot(xs, [r["val_acc1"] for r in records], label="val@1")
        axes[1, 0].plot(xs, [r["train_acc5"] for r in records], label="train@5")
        axes[1, 0].plot(xs, [r["val_acc5"] for r in records], label="val@5")
        axes[1, 1].plot(xs, [r["lr"] for r in records])
        for axis, title in zip(axes.flat, ["Loss", "Top-1", "Top-5", "Learning rate"]):
            axis.set_title(title)
            axis.set_xlabel("epoch")
            if axis.lines and title != "Learning rate":
                axis.legend()
        fig.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def capture_rng():
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device_type == "cuda" else None,
        }

    def collect_rng():
        state = capture_rng()
        if not distributed:
            return [state]
        gathered = [None] * world_size if is_main() else None
        dist.gather_object(state, gathered, dst=0)
        return gathered

    def restore_rng(checkpoint):
        states = checkpoint.get("rng_by_rank")
        if not states:
            log("warning: checkpoint has no per-rank RNG state")
            return
        if len(states) != world_size:
            raise ValueError(
                f"Resume world size mismatch: checkpoint={len(states)}, current={world_size}"
            )
        state = states[rank]
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if device_type == "cuda" and state.get("cuda") is not None:
            torch.cuda.set_rng_state(state["cuda"], device=device)

    def scheduler_for(optimizer, total_steps, steps_per_epoch):
        warmup_steps = warmup_epochs * steps_per_epoch
        min_factor = min_lr / max_lr
        decay_steps = max(1, total_steps - warmup_steps)

        def factor(step):
            if warmup_steps and step < warmup_steps:
                return min_factor + (1.0 - min_factor) * step / warmup_steps
            fraction = min(1.0, (step - warmup_steps) / decay_steps)
            return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * fraction))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

    def validate_resume(checkpoint, seed, params):
        if int(checkpoint.get("seed", seed)) != seed:
            raise ValueError("Resume seed mismatch")
        if int(checkpoint.get("world_size", world_size)) != world_size:
            raise ValueError(
                f"Resume world size mismatch: checkpoint={checkpoint.get('world_size')}, current={world_size}"
            )
        if int(checkpoint.get("steps_per_epoch", len(train_loader))) != len(train_loader):
            raise ValueError("Resume train-loader length mismatch")
        current = config(seed, params)
        saved = checkpoint.get("config", {})
        critical = (
            "model_family", "model_arch", "block_type", "mode", "block_depth", "stem",
            "frontend", "block", "normalization_state", "width", "feature_resolution",
            "weight_tying", "T", "activation_checkpointing", "img", "resize", "epochs",
            "batch_size", "workers", "limit_train", "limit_val", "max_lr", "min_lr",
            "warmup_epochs", "grad_accum_steps", "optimizer", "weight_decay",
            "label_smoothing", "weight_decay_filter", "amp", "amp_dtype", "num_classes", "parameters",
        )
        mismatches = [
            f"{key}: checkpoint={saved[key]!r}, current={current[key]!r}"
            for key in critical if key in saved and saved[key] != current[key]
        ]
        if mismatches:
            raise ValueError("Resume configuration mismatch:\n  " + "\n  ".join(mismatches))

    def save_checkpoint(path, model, optimizer, scheduler, epoch, best, val1, val5,
                        seed, params, history, rng_by_rank):
        if not is_main():
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "format_version": 3,
            "epoch": epoch,
            "global_step": scheduler.last_epoch,
            "best_val1": best,
            "val1": val1,
            "val5": val5,
            "seed": seed,
            "parameters": params,
            "params_m": params / 1e6,
            "model": unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config(seed, params),
            "history": history,
            "world_size": world_size,
            "steps_per_epoch": len(train_loader),
            "rng_by_rank": rng_by_rank,
        }
        temporary = f"{path}.tmp"
        torch.save(state, temporary)
        os.replace(temporary, path)

    def evaluate(model, loss_fn):
        model.eval()
        totals = [0.0, 0, 0, 0]
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast_ctx():
                    outputs, _ = model(inputs)
                    loss = loss_fn(outputs, targets)
                count = targets.numel()
                totals[0] += loss.item() * count
                totals[1] += (outputs.argmax(1) == targets).sum().item()
                totals[2] += outputs.topk(min(5, num_classes), 1).indices.eq(targets[:, None]).any(1).sum().item()
                totals[3] += count
        loss_sum, top1, top5, count = metric_sums(totals)
        return loss_sum / count, top1 / count, top5 / count

    def run_memory_probe():
        if device_type != "cuda":
            raise RuntimeError("MEMORY_PROBE=1 requires CUDA")
        model = RecurrentCNN(
            block_type, iterations, num_classes, mode, block_depth
        ).to(device).train()
        optimizer = torch.optim.AdamW(
            adamw_parameter_groups(model, 0.05), lr=max_lr, weight_decay=0.0
        )
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        inputs, targets = next(iter(train_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with autocast_ctx():
            outputs, _ = model(inputs)
            loss = loss_fn(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        log(
            f"memory_probe block_type={block_type} batch_size={targets.numel()} "
            f"peak_allocated_gib={torch.cuda.max_memory_allocated(device) / 1024**3:.3f} "
            f"peak_reserved_gib={torch.cuda.max_memory_reserved(device) / 1024**3:.3f}"
        )

    def train(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = RecurrentCNN(
            block_type, iterations, num_classes, mode, block_depth
        ).to(device)
        params = sum(parameter.numel() for parameter in model.parameters())
        run_dir = os.path.join(output_dir, f"{block_type}_seed{seed}")
        history_path = os.path.join(run_dir, "history.json")
        curves_path = os.path.join(run_dir, "curves.png")
        history = {"config": config(seed, params), "epochs": []}

        if resume_checkpoint is not None:
            validate_resume(resume_checkpoint, seed, params)
            model.load_state_dict(resume_checkpoint["model"], strict=True)
        if distributed:
            model = DDP(
                model,
                device_ids=[local_rank] if device_type == "cuda" else None,
                find_unused_parameters=False,
            )
        optimizer = torch.optim.AdamW(
            adamw_parameter_groups(model, 0.05), lr=max_lr, weight_decay=0.0
        )
        optimizer_steps_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
        scheduler = scheduler_for(optimizer, epochs * optimizer_steps_per_epoch, optimizer_steps_per_epoch)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        start_epoch, best, final_acc1, final_acc5 = 0, -1.0, 0.0, 0.0

        if resume_checkpoint is not None:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
            if resume_checkpoint.get("scaler"):
                scaler.load_state_dict(resume_checkpoint["scaler"])
            start_epoch = int(resume_checkpoint["epoch"]) + 1
            best = float(resume_checkpoint.get("best_val1", -1.0))
            final_acc1 = float(resume_checkpoint.get("val1", 0.0))
            final_acc5 = float(resume_checkpoint.get("val5", 0.0))
            old_history = resume_checkpoint.get("history")
            if isinstance(old_history, dict):
                history = old_history
                history["epochs"] = [
                    record for record in history.get("epochs", [])
                    if int(record["epoch"]) < start_epoch
                ]
                history["config"] = config(seed, params)
            restore_rng(resume_checkpoint)
            log(
                f"resumed={resume_path} completed_epoch={start_epoch - 1} "
                f"next_epoch={start_epoch} global_step={scheduler.last_epoch}"
            )

        elapsed_offset = history["epochs"][-1].get("elapsed_sec", 0.0) if history["epochs"] else 0.0
        started = time.time() - elapsed_offset
        for epoch in range(start_epoch, epochs):
            epoch_started = time.time()
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            totals = [0.0, 0, 0, 0]
            optimizer.zero_grad(set_to_none=True)
            iterable = train_loader
            if tqdm is not None:
                iterable = tqdm(
                    train_loader,
                    desc=f"{block_type} s{seed} ep {epoch + 1}/{epochs}",
                    disable=(not progress) or (not is_main()),
                    dynamic_ncols=True,
                    leave=False,
                )
            for batch_index, (inputs, targets) in enumerate(iterable):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                accumulation_start = (batch_index // grad_accum_steps) * grad_accum_steps
                accumulation_size = min(grad_accum_steps, len(train_loader) - accumulation_start)
                update = batch_index + 1 == len(train_loader) or (batch_index + 1) % grad_accum_steps == 0
                sync_ctx = model.no_sync() if isinstance(model, DDP) and not update else nullcontext()
                with sync_ctx:
                    with autocast_ctx():
                        outputs, _ = model(inputs)
                        loss = loss_fn(outputs, targets)
                    scaler.scale(loss / accumulation_size).backward()
                count = targets.numel()
                totals[0] += loss.detach().item() * count
                totals[1] += (outputs.argmax(1) == targets).sum().item()
                totals[2] += outputs.topk(min(5, num_classes), 1).indices.eq(targets[:, None]).any(1).sum().item()
                totals[3] += count
                if update:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if is_main() and hasattr(iterable, "set_postfix"):
                    iterable.set_postfix(loss=f"{loss.item():.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            loss_sum, top1, top5, count = metric_sums(totals)
            train_loss, train_acc1, train_acc5 = loss_sum / count, top1 / count, top5 / count
            val_loss, final_acc1, final_acc5 = evaluate(model, loss_fn)
            improved = final_acc1 > best
            best = max(best, final_acc1)
            latest = os.path.join(run_dir, "checkpoint_latest.pt")
            best_path = os.path.join(run_dir, "checkpoint_best.pt")
            final_path = os.path.join(run_dir, "checkpoint_final.pt")
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc1": train_acc1,
                "train_acc5": train_acc5,
                "val_loss": val_loss,
                "val_acc1": final_acc1,
                "val_acc5": final_acc5,
                "best_val1": best,
                "lr": scheduler.get_last_lr()[0],
                "epoch_sec": time.time() - epoch_started,
                "elapsed_sec": time.time() - started,
                "checkpoint_latest": latest,
                "checkpoint_best": best_path if improved else None,
                "checkpoint_final": final_path if epoch + 1 == epochs else None,
            }
            history["epochs"].append(record)
            log(
                f"[{block_type} s{seed}] ep{epoch:2d} train_loss={train_loss:.3f} "
                f"train@1={train_acc1:.3f} val_loss={val_loss:.3f} "
                f"val@1={final_acc1:.3f} val@5={final_acc5:.3f} best@1={best:.3f}"
            )
            save_json(history_path, history)
            save_curves(curves_path, history["epochs"])
            rng_by_rank = collect_rng()
            save_checkpoint(latest, model, optimizer, scheduler, epoch, best, final_acc1,
                            final_acc5, seed, params, history, rng_by_rank)
            if improved:
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, best,
                                final_acc1, final_acc5, seed, params, history, rng_by_rank)
            if epoch + 1 == epochs:
                save_checkpoint(final_path, model, optimizer, scheduler, epoch, best,
                                final_acc1, final_acc5, seed, params, history, rng_by_rank)

        final_path = os.path.join(run_dir, "checkpoint_final.pt")
        if start_epoch >= epochs and not os.path.isfile(final_path):
            rng_by_rank = collect_rng()
            save_checkpoint(final_path, model, optimizer, scheduler, epochs - 1, best,
                            final_acc1, final_acc5, seed, params, history, rng_by_rank)
        model.eval()
        with torch.no_grad():
            inputs, _ = next(iter(val_loader))
            with autocast_ctx():
                _, residuals = model(inputs.to(device, non_blocking=True), log_residuals=True)
        return {
            "seed": seed,
            "parameters": params,
            "params_m": params / 1e6,
            "best_val1": best,
            "residual_by_iteration": residuals,
        }

    log(f"data={data_root} classes={num_classes} train_batches={len(train_loader)} val_batches={len(val_loader)}")
    log(f"device={device} distributed={distributed} world_size={world_size} amp={amp_enabled} amp_dtype={amp_dtype_name}")
    log(
        f"block_type={block_type} mode={mode} block_depth={block_depth} "
        f"native_width={arch['width']} feature_resolution={arch['feature_resolution']} "
        f"T={iterations} tied_cell=True "
        f"normalization_state={arch['normalization_state']} "
        "weight_decay_filter=no_decay_for_1d_parameters_and_bias "
        "activation_checkpointing=False"
    )
    log(
        f"epochs={epochs} BS_per_gpu={batch_size} grad_accum_steps={grad_accum_steps} "
        f"effective_batch_size={batch_size * world_size * grad_accum_steps} "
        f"lr={max_lr:g}->{min_lr:g} warmup_epochs={warmup_epochs} output_dir={output_dir}"
    )

    if memory_probe:
        run_memory_probe()
    else:
        results = [train(seed) for seed in seeds]
        if is_main():
            summary_path = os.path.join(output_dir, f"{block_type}_summary.json")
            save_json(summary_path, {
                "config": config(seeds[0], results[0]["parameters"]),
                "runs": results,
            })
            log(f"summary={summary_path}")
            for result in results:
                log(
                    f"seed={result['seed']} params={result['parameters']:,} "
                    f"best_val1={result['best_val1']:.6f} "
                    f"residual_by_iteration={result['residual_by_iteration']}"
                )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Python exceptions (including loader timeouts) otherwise bypass the
        # normal cleanup at the end of main and leave a noisy NCCL warning.
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
