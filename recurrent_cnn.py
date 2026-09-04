"""Four-stage, weight-tied ConvNeXt V1/V2 baselines for ImageNet-1K.

ARR1 selects the number of unique CNBlocks in each native ConvNeXt stage. ARR2
selects how many times the complete stage is applied with the same parameters.
Enabled stages must form a contiguous prefix of the four native stage widths.
V selects the ConvNeXt block generation: 1 for V1 and 2 for V2 with GRN.
"""

from collections import OrderedDict
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models.convnext import CNBlock, LayerNorm2d
from torchvision.ops import stochastic_depth

from imagenet_data import make_imagenet_loaders
from deltanet import DeltaNet
from rats_attention import RATSAttention

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


EXPERIMENT_VERSION = 8
STAGE_WIDTHS = (96, 192, 384, 768)
DEFAULT_STAGE_DEPTHS = (1, 1, 1, 0)
DEFAULT_STAGE_REPEATS = (12, 12, 12, 0)
DEFAULT_REG_MODE = (0, 0, 0, 0)
DEFAULT_N_REG = (8, 8, 8, 8)
REG_HEADS = 6
REG_SDPA_BACKEND = "flash"
DELTA_BACKEND = "auto"
DELTA_CHUNK_SIZE = 64
DELTA_CONV_SIZE = 4
DELTA_NORM_EPS = 1e-5
SUPPORTED_CONVNEXT_VERSIONS = (1, 2)


def env_flag(name, default=0):
    return bool(int(os.environ.get(name, default)))


def validate_convnext_version(value):
    """Normalize the user-facing V selector."""
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V must be 1 or 2, got {value!r}") from exc
    if version not in SUPPORTED_CONVNEXT_VERSIONS:
        raise ValueError(f"V must be 1 or 2, got {version}")
    return version


def parse_stage_array(value, name):
    """Parse one four-integer comma-separated architecture array."""
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value
    try:
        parsed = tuple(int(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain four comma-separated integers") from exc
    return parsed


def validate_register_arrays(
    reg_mode=DEFAULT_REG_MODE,
    n_reg=DEFAULT_N_REG,
    stage_depths=None,
):
    """Normalize per-stage register flags/counts and validate active stages."""
    modes = parse_stage_array(reg_mode, "REG_MODE")
    counts = parse_stage_array(n_reg, "N_REG")
    if len(modes) != 4 or len(counts) != 4:
        raise ValueError(
            "REG_MODE and N_REG must each contain exactly four integers; "
            f"got {len(modes)} and {len(counts)}"
        )
    if any(value not in {0, 1} for value in modes):
        raise ValueError(f"REG_MODE values must be 0 or 1, got {modes}")
    if any(value < 1 for value in counts):
        raise ValueError(f"N_REG values must be positive, got {counts}")
    if stage_depths is not None:
        depths = parse_stage_array(stage_depths, "ARR1")
        if len(depths) != 4:
            raise ValueError(f"ARR1 must contain exactly four integers, got {len(depths)}")
        invalid = [
            index + 1
            for index, (mode, depth) in enumerate(zip(modes, depths))
            if mode and depth == 0
        ]
        if invalid:
            raise ValueError(
                "REG_MODE cannot enable inactive stages: "
                + ", ".join(map(str, invalid))
            )
    return modes, counts


def model_arch_name(
    version,
    reg_mode=DEFAULT_REG_MODE,
    delta_mode=False,
    reg_head=False,
):
    version = validate_convnext_version(version)
    modes, _ = validate_register_arrays(reg_mode, DEFAULT_N_REG)
    if delta_mode or reg_head:
        variants = []
        if delta_mode:
            variants.append("delta")
        if reg_head:
            variants.append("reghead")
        suffix = f"four_stage_array_tied_rats_{'_'.join(variants)}_v8"
    else:
        suffix = "four_stage_array_tied_rats_v7" if any(modes) else "four_stage_array_tied_v6"
    return f"recurrent_convnext_v{version}_{suffix}"


def validate_stage_arrays(stage_depths, stage_repeats):
    """Return normalized arrays after enforcing a contiguous active prefix."""
    depths = parse_stage_array(stage_depths, "ARR1")
    repeats = parse_stage_array(stage_repeats, "ARR2")
    if len(depths) != 4 or len(repeats) != 4:
        raise ValueError(
            f"ARR1 and ARR2 must each contain exactly four integers; "
            f"got {len(depths)} and {len(repeats)}"
        )
    if any(value < 0 for value in depths + repeats):
        raise ValueError("ARR1 and ARR2 values must be non-negative")

    disabled = False
    active_stages = 0
    for index, (depth, repeat) in enumerate(zip(depths, repeats), start=1):
        if (depth == 0) != (repeat == 0):
            raise ValueError(
                f"Stage {index} must have both ARR1 and ARR2 positive or both zero; "
                f"got depth={depth}, repeats={repeat}"
            )
        if depth == 0:
            disabled = True
        else:
            if disabled:
                raise ValueError("Enabled stages must form a contiguous prefix")
            active_stages += 1
    if active_stages == 0:
        raise ValueError("At least stage 1 must be enabled")
    return depths, repeats


def legacy_config_to_stage_arrays(config):
    """Translate v2-v5 MODE/T/BLOCK_DEPTH metadata into ARR1/ARR2."""
    if "arr1" in config and "arr2" in config:
        return validate_stage_arrays(config["arr1"], config["arr2"])

    mode = str(config.get("mode", "naive")).lower()
    depth = int(config.get("block_depth", 1))
    iterations = int(config.get("T", 12))
    if depth < 1 or iterations < 1:
        raise ValueError(
            f"Invalid legacy architecture: block_depth={depth}, T={iterations}"
        )
    mappings = {
        "naive": ((depth, 0, 0, 0), (iterations, 0, 0, 0)),
        "pro": ((3, 3, depth, 0), (1, 1, iterations, 0)),
        "promax": ((3, 3, depth, 0), (iterations, iterations, iterations, 0)),
        "promini": ((1, 1, depth, 0), (iterations, iterations, iterations, 0)),
    }
    if mode not in mappings:
        raise ValueError(f"Unsupported legacy MODE={mode!r}")
    return validate_stage_arrays(*mappings[mode])


class ScheduledCNBlock(CNBlock):
    """Torchvision CNBlock with a per-call stochastic-depth probability."""

    def forward(self, inputs, stochastic_depth_prob=None):
        if stochastic_depth_prob is None:
            stochastic_depth_prob = getattr(self, "_drop_path_prob", 0.0)
        outputs = self.layer_scale * self.block(inputs)
        outputs = stochastic_depth(
            outputs,
            float(stochastic_depth_prob),
            "row",
            self.training,
        )
        return inputs + outputs


class RATSRegisterBlock(nn.Module):
    """PEQ RATS update applied after one complete ConvNeXt stage execution."""

    def __init__(
        self,
        width,
        n_reg,
        heads=REG_HEADS,
        sdpa_backend=REG_SDPA_BACKEND,
        delta_mode=False,
    ):
        super().__init__()
        if n_reg < 1:
            raise ValueError(f"n_reg={n_reg} must be positive")
        self.width = int(width)
        self.n_reg = int(n_reg)
        self.delta_mode = bool(delta_mode)
        self.r0 = nn.Parameter(torch.randn(1, self.n_reg, self.width) * 0.02)
        if self.delta_mode:
            self.lnp = nn.LayerNorm(self.width, eps=DELTA_NORM_EPS)
            self.patch_delta = DeltaNet(
                self.width,
                heads,
                conv_size=DELTA_CONV_SIZE,
                norm_eps=DELTA_NORM_EPS,
                backend=DELTA_BACKEND,
                chunk_size=DELTA_CHUNK_SIZE,
            )
        self.lnc = nn.LayerNorm(self.width, eps=1e-5)
        self.lnr = nn.LayerNorm(self.width, eps=1e-5)
        self.lnb = nn.LayerNorm(self.width, eps=1e-5)
        self.lnm = nn.LayerNorm(self.width, eps=1e-5)
        self.attention = RATSAttention(
            self.width,
            heads=heads,
            sdpa_backend=sdpa_backend,
        )
        self.w1 = nn.Linear(self.width, 4 * self.width)
        self.act = nn.GELU()
        self.w2 = nn.Linear(4 * self.width, self.width)

    def initial_registers(self, batch_size):
        return self.r0.expand(batch_size, -1, -1).contiguous()

    def forward(self, inputs, registers):
        if inputs.ndim != 4 or inputs.size(1) != self.width:
            raise ValueError(
                f"Expected NCHW features with C={self.width}, got {tuple(inputs.shape)}"
            )
        if registers.shape != (inputs.size(0), self.n_reg, self.width):
            raise ValueError(
                "Register shape mismatch: expected "
                f"{(inputs.size(0), self.n_reg, self.width)}, got {tuple(registers.shape)}"
            )
        batch, channels, height, width = inputs.shape
        tokens = inputs.flatten(2).transpose(1, 2)
        if self.delta_mode:
            tokens = tokens + self.patch_delta(self.lnp(tokens))

        features = self.lnc(tokens)
        registers = registers + self.attention(
            "compress", self.lnr(registers), features, features
        )
        normalized_registers = self.lnr(registers)
        registers = registers + self.attention(
            "refine",
            normalized_registers,
            normalized_registers,
            normalized_registers,
        )
        normalized_registers = self.lnr(registers)
        tokens = tokens + self.attention(
            "broadcast",
            self.lnb(tokens),
            normalized_registers,
            normalized_registers,
        )
        tokens = tokens + self.w2(self.act(self.w1(self.lnm(tokens))))
        outputs = tokens.transpose(1, 2).contiguous().reshape(
            batch, channels, height, width
        )
        return outputs, registers


class RepeatedStage(nn.Module):
    """Apply one shape-preserving stage repeatedly with shared parameters."""

    def __init__(self, stage, repeats, drop_path_probs=None, register_block=None):
        super().__init__()
        if repeats < 1:
            raise ValueError(f"repeats={repeats} must be positive")
        self.stage = stage
        self.repeats = repeats
        self.register_block = register_block
        applications = repeats * len(stage)
        if drop_path_probs is None:
            drop_path_probs = (0.0,) * applications
        self.drop_path_probs = tuple(float(value) for value in drop_path_probs)
        if len(self.drop_path_probs) != applications:
            raise ValueError(
                "drop_path_probs must contain one value per block application; "
                f"expected {applications}, got {len(self.drop_path_probs)}"
            )

    def forward(self, inputs, log_residuals=False):
        outputs = inputs
        registers = (
            self.register_block.initial_registers(inputs.size(0))
            if self.register_block is not None
            else None
        )
        residuals = []
        application_index = 0
        for _ in range(self.repeats):
            previous = outputs
            for block in self.stage:
                block._drop_path_prob = self.drop_path_probs[application_index]
                application_index += 1
            outputs = self.stage(outputs)
            if self.register_block is not None:
                outputs, registers = self.register_block(outputs, registers)
            if log_residuals:
                numerator = (outputs - previous).float().flatten(1).norm(dim=1).mean()
                denominator = outputs.float().flatten(1).norm(dim=1).mean().clamp_min(1e-8)
                residuals.append((numerator / denominator).item())
        return outputs, registers, residuals


class GlobalResponseNorm(nn.Module):
    """ConvNeXt V2 Global Response Normalization for NHWC activations."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.eps = eps

    def forward(self, inputs):
        response = torch.norm(inputs, p=2, dim=(1, 2), keepdim=True)
        normalized = response / (response.mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (inputs * normalized) + self.beta + inputs


class ConvNeXtV2Block(nn.Module):
    """Official ConvNeXt V2 block with a per-call stochastic-depth probability."""

    def __init__(self, width):
        super().__init__()
        self.dwconv = nn.Conv2d(
            width, width, kernel_size=7, padding=3, groups=width
        )
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.pwconv1 = nn.Linear(width, 4 * width)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(4 * width)
        self.pwconv2 = nn.Linear(4 * width, width)

    def forward(self, inputs, stochastic_depth_prob=None):
        if stochastic_depth_prob is None:
            stochastic_depth_prob = getattr(self, "_drop_path_prob", 0.0)
        outputs = self.dwconv(inputs)
        outputs = outputs.permute(0, 2, 3, 1)
        outputs = self.norm(outputs)
        outputs = self.pwconv1(outputs)
        outputs = self.act(outputs)
        outputs = self.grn(outputs)
        outputs = self.pwconv2(outputs)
        outputs = outputs.permute(0, 3, 1, 2)
        outputs = stochastic_depth(
            outputs,
            float(stochastic_depth_prob),
            "row",
            self.training,
        )
        return inputs + outputs


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
    """A four-stage ConvNeXt with an independently repeated stage at each width."""

    def __init__(
        self,
        stage_depths=DEFAULT_STAGE_DEPTHS,
        stage_repeats=DEFAULT_STAGE_REPEATS,
        num_classes=1000,
        convnext_version=1,
        drop_path_rate=0.0,
        reg_mode=DEFAULT_REG_MODE,
        n_reg=DEFAULT_N_REG,
        delta_mode=False,
        reg_head=False,
    ):
        super().__init__()
        depths, repeats = validate_stage_arrays(stage_depths, stage_repeats)
        reg_mode, n_reg = validate_register_arrays(reg_mode, n_reg, depths)
        self.convnext_version = validate_convnext_version(convnext_version)
        self.stage_depths = depths
        self.stage_repeats = repeats
        self.reg_mode = reg_mode
        self.n_reg = n_reg
        self.delta_mode = bool(delta_mode)
        self.reg_head = bool(reg_head)
        self.active_stages = sum(depth > 0 for depth in depths)
        self.last_width = STAGE_WIDTHS[self.active_stages - 1]
        self.unique_blocks = sum(depths)
        self.block_applications = sum(
            depth * repeat for depth, repeat in zip(depths, repeats)
        )
        self.register_stage_count = sum(reg_mode)
        self.register_applications = sum(
            repeat for mode, repeat in zip(reg_mode, repeats) if mode
        )
        if self.delta_mode and not any(reg_mode):
            raise ValueError("delta_mode=True requires at least one enabled register stage")
        if self.reg_head and not reg_mode[self.active_stages - 1]:
            raise ValueError(
                "reg_head=True requires registers in the final active stage"
            )
        self.drop_path_rate = float(drop_path_rate)
        if not 0.0 <= self.drop_path_rate < 1.0:
            raise ValueError("drop_path_rate must be in [0, 1)")
        if self.block_applications == 1:
            self.drop_path_rates = (0.0,)
        else:
            denominator = self.block_applications - 1
            self.drop_path_rates = tuple(
                self.drop_path_rate * index / denominator
                for index in range(self.block_applications)
            )
        self.drop_path_schedule = "unrolled_linear"

        stem_width = 96
        stem_norm = partial(LayerNorm2d, eps=1e-6)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_width, kernel_size=4, stride=4, bias=True),
            stem_norm(stem_width),
        )

        # Interleaving stages and downsampling modules preserves the legacy
        # optimizer parameter order after deterministic state_dict key migration.
        features = []
        self.stage_feature_indices = []
        drop_path_offset = 0
        for stage_index in range(self.active_stages):
            width = STAGE_WIDTHS[stage_index]
            stage = nn.Sequential(
                *(
                    self._make_convnext_block(width, self.convnext_version)
                    for _ in range(depths[stage_index])
                )
            )
            stage_applications = depths[stage_index] * repeats[stage_index]
            stage_drop_path_probs = self.drop_path_rates[
                drop_path_offset:drop_path_offset + stage_applications
            ]
            drop_path_offset += stage_applications
            self.stage_feature_indices.append(len(features))
            register_block = (
                RATSRegisterBlock(
                    width,
                    n_reg[stage_index],
                    delta_mode=self.delta_mode,
                )
                if reg_mode[stage_index]
                else None
            )
            features.append(
                RepeatedStage(
                    stage,
                    repeats[stage_index],
                    drop_path_probs=stage_drop_path_probs,
                    register_block=register_block,
                )
            )
            if stage_index + 1 < self.active_stages:
                features.append(self._make_downsample(width, STAGE_WIDTHS[stage_index + 1]))
        self.features = nn.ModuleList(features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.readout_width = self.last_width * (2 if self.reg_head else 1)
        self.head_norm = nn.LayerNorm(self.readout_width, eps=1e-6)
        self.head = nn.Linear(self.readout_width, num_classes)
        self.apply(self._init_convnext)

    @staticmethod
    def _make_convnext_block(width, convnext_version):
        if convnext_version == 2:
            return ConvNeXtV2Block(width)
        return ScheduledCNBlock(
            width,
            layer_scale=1e-6,
            stochastic_depth_prob=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

    @staticmethod
    def _make_downsample(in_width, out_width):
        norm = partial(LayerNorm2d, eps=1e-6)
        return nn.Sequential(
            norm(in_width),
            nn.Conv2d(in_width, out_width, kernel_size=2, stride=2),
        )

    @staticmethod
    def _init_convnext(module):
        # Matches torchvision.models.convnext.ConvNeXt.__init__.
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, log_residuals=False):
        x = self.stem(x)
        residuals = {}
        stage_number = 0
        final_registers = None
        for module in self.features:
            if isinstance(module, RepeatedStage):
                stage_number += 1
                x, final_registers, stage_residuals = module(
                    x, log_residuals=log_residuals
                )
                if log_residuals:
                    residuals[f"stage{stage_number}"] = stage_residuals
            else:
                x = module(x)
        x = self.pool(x).flatten(1)
        if self.reg_head:
            if final_registers is None:
                raise RuntimeError("Final-stage registers are unavailable for reg_head")
            x = torch.cat((final_registers.mean(dim=1), x), dim=-1)
        return self.head(self.head_norm(x)), residuals


def migrate_legacy_state_dict(state_dict, config):
    """Map v2-v5 ConvNeXt parameter names onto the interleaved v6 features."""
    if int(config.get("experiment_version", 0)) >= 6:
        return state_dict

    mode = str(config.get("mode", "naive")).lower()
    block_depth = int(config.get("block_depth", 1))
    if mode not in {"naive", "pro", "promax", "promini"}:
        raise ValueError(f"Unsupported legacy MODE={mode!r}")

    def map_final_stage(remainder, feature_index):
        if block_depth == 1:
            remainder = f"0.{remainder}"
        return f"features.{feature_index}.stage.{remainder}"

    migrated = OrderedDict()
    for key, value in state_dict.items():
        new_key = None
        if key.startswith(("stem.", "head_norm.", "head.")):
            new_key = key
        elif mode == "naive" and key.startswith("block."):
            new_key = map_final_stage(key[len("block."):], 0)
        elif mode in {"pro", "promax", "promini"}:
            if mode == "pro":
                stage1_prefix = "frontend.0."
                stage2_prefix = "frontend.2."
            else:
                stage1_prefix = "frontend.0.stage."
                stage2_prefix = "frontend.2.stage."
            if key.startswith(stage1_prefix):
                new_key = f"features.0.stage.{key[len(stage1_prefix):]}"
            elif key.startswith("frontend.1."):
                new_key = f"features.1.{key[len('frontend.1.'):]}"
            elif key.startswith(stage2_prefix):
                new_key = f"features.2.stage.{key[len(stage2_prefix):]}"
            elif key.startswith("frontend.3."):
                new_key = f"features.3.{key[len('frontend.3.'):]}"
            elif key.startswith("block."):
                new_key = map_final_stage(key[len("block."):], 4)
        if new_key is None:
            raise ValueError(f"Cannot migrate legacy model key {key!r} for mode={mode}")
        if new_key in migrated:
            raise ValueError(f"Legacy state migration produced duplicate key {new_key!r}")
        migrated[new_key] = value
    return migrated


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
    if checkpoint_version not in {2, 3, 4, 5, 6, 7, 8}:
        raise ValueError(
            "This checkpoint predates the supported ConvNeXt experiment versions "
            "and cannot be resumed exactly. Start a new run instead."
        )
    stage_depths, stage_repeats = legacy_config_to_stage_arrays(config)
    convnext_version = validate_convnext_version(config.get("convnext_version", 1))
    reg_mode, n_reg = validate_register_arrays(
        config.get("reg_mode", DEFAULT_REG_MODE),
        config.get("n_reg", DEFAULT_N_REG),
        stage_depths,
    )
    delta_mode = bool(config.get("delta_mode", False))
    reg_head = bool(config.get("reg_head", False))
    os.environ["ARR1"] = ",".join(str(value) for value in stage_depths)
    os.environ["ARR2"] = ",".join(str(value) for value in stage_repeats)
    os.environ["V"] = str(convnext_version)
    os.environ["REG_MODE"] = ",".join(str(value) for value in reg_mode)
    os.environ["N_REG"] = ",".join(str(value) for value in n_reg)
    os.environ["DELTA_MODE"] = str(int(delta_mode))
    os.environ["REG_HEAD"] = str(int(reg_head))
    checkpoint["model"] = migrate_legacy_state_dict(checkpoint["model"], config)
    checkpoint["stage_depths"] = stage_depths
    checkpoint["stage_repeats"] = stage_repeats
    checkpoint["convnext_version"] = convnext_version
    checkpoint["reg_mode"] = reg_mode
    checkpoint["n_reg"] = n_reg
    checkpoint["delta_mode"] = delta_mode
    checkpoint["reg_head"] = reg_head
    checkpoint["legacy_experiment_version"] = checkpoint_version
    env_from_config = {
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
    stage_depths, stage_repeats = validate_stage_arrays(
        os.environ.get("ARR1", ",".join(map(str, DEFAULT_STAGE_DEPTHS))),
        os.environ.get("ARR2", ",".join(map(str, DEFAULT_STAGE_REPEATS))),
    )
    reg_mode, n_reg = validate_register_arrays(
        os.environ.get("REG_MODE", ",".join(map(str, DEFAULT_REG_MODE))),
        os.environ.get("N_REG", ",".join(map(str, DEFAULT_N_REG))),
        stage_depths,
    )
    delta_mode = env_flag("DELTA_MODE", 0)
    reg_head = env_flag("REG_HEAD", 0)
    convnext_version = validate_convnext_version(os.environ.get("V", 1))
    model_arch = model_arch_name(
        convnext_version,
        reg_mode,
        delta_mode=delta_mode,
        reg_head=reg_head,
    )
    active_stages = sum(depth > 0 for depth in stage_depths)
    last_width = STAGE_WIDTHS[active_stages - 1]
    unique_blocks = sum(stage_depths)
    block_applications = sum(
        depth * repeat for depth, repeat in zip(stage_depths, stage_repeats)
    )
    register_stage_count = sum(reg_mode)
    register_applications = sum(
        repeat for mode, repeat in zip(reg_mode, stage_repeats) if mode
    )

    if env_flag("PRINT_MODEL_INFO", 0):
        model = RecurrentCNN(
            stage_depths,
            stage_repeats,
            num_classes=1000,
            convnext_version=convnext_version,
            reg_mode=reg_mode,
            n_reg=n_reg,
            delta_mode=delta_mode,
            reg_head=reg_head,
        )
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(json.dumps({
            "experiment_version": EXPERIMENT_VERSION,
            "model_arch": model_arch,
            "convnext_version": convnext_version,
            "arr1": list(stage_depths),
            "arr2": list(stage_repeats),
            "reg_mode": list(reg_mode),
            "n_reg": list(n_reg),
            "delta_mode": delta_mode,
            "reg_head": reg_head,
            "delta_backend": DELTA_BACKEND if delta_mode else None,
            "delta_chunk_size": DELTA_CHUNK_SIZE if delta_mode else None,
            "delta_conv_size": DELTA_CONV_SIZE if delta_mode else None,
            "delta_norm_eps": DELTA_NORM_EPS if delta_mode else None,
            "register_readout": "concat_mean_reg_feature" if reg_head else "feature_mean",
            "stage_widths": list(STAGE_WIDTHS),
            "active_stages": active_stages,
            "last_width": last_width,
            "feature_resolution_at_224": 56 // (2 ** (active_stages - 1)),
            "unique_blocks": unique_blocks,
            "block_applications": block_applications,
            "register_stage_count": register_stage_count,
            "register_applications": register_applications,
            "register_attention": "rats_shared_qkv_identity_registers",
            "register_heads": REG_HEADS,
            "register_sdpa_backend": REG_SDPA_BACKEND,
            "register_mlp_ratio": 4,
            "parameters": total,
            "trainable_parameters": trainable,
            "stochastic_depth_prob": 0.0,
            "normalization_state": "stateless_layernorm_per_call",
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

    arr1_slug = "-".join(str(value) for value in stage_depths)
    arr2_slug = "-".join(str(value) for value in stage_repeats)
    reg_mode_slug = "-".join(str(value) for value in reg_mode)
    n_reg_slug = "-".join(str(value) for value in n_reg)
    register_slug = (
        f"_REG-{reg_mode_slug}_NREG-{n_reg_slug}" if register_stage_count else ""
    )
    variant_slug = (
        ("_DELTA1" if delta_mode else "")
        + ("_REGHEAD1" if reg_head else "")
    )
    register_slug += variant_slug
    output_experiment_version = (
        EXPERIMENT_VERSION
        if delta_mode or reg_head
        else (7 if register_stage_count else 6)
    )
    output_dir = os.environ.get(
        "OUTPUT_DIR",
        f"outputs/imagenet_recurrent_v{output_experiment_version}_convnextV{convnext_version}_"
        f"ARR1-{arr1_slug}_ARR2-{arr2_slug}{register_slug}_img{img}_epochs{epochs}_"
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
            "experiment_version": EXPERIMENT_VERSION,
            "model_arch": model_arch,
            "convnext_version": convnext_version,
            "arr1": list(stage_depths),
            "arr2": list(stage_repeats),
            "reg_mode": list(reg_mode),
            "n_reg": list(n_reg),
            "delta_mode": delta_mode,
            "reg_head": reg_head,
            "delta_backend": DELTA_BACKEND if delta_mode else None,
            "delta_chunk_size": DELTA_CHUNK_SIZE if delta_mode else None,
            "delta_conv_size": DELTA_CONV_SIZE if delta_mode else None,
            "delta_norm_eps": DELTA_NORM_EPS if delta_mode else None,
            "register_readout": "concat_mean_reg_feature" if reg_head else "feature_mean",
            "stage_widths": list(STAGE_WIDTHS),
            "active_stages": active_stages,
            "last_width": last_width,
            "feature_resolution": img // (4 * 2 ** (active_stages - 1)),
            "unique_blocks": unique_blocks,
            "block_applications": block_applications,
            "register_stage_count": register_stage_count,
            "register_applications": register_applications,
            "register_attention": "rats_shared_qkv_identity_registers",
            "register_heads": REG_HEADS,
            "register_sdpa_backend": REG_SDPA_BACKEND,
            "register_mlp_ratio": 4,
            "register_data_term": False,
            "register_reconstruction": False,
            "register_layerscale": False,
            "stem": "convnext_tiny_conv4_s4_layernorm2d",
            "normalization_state": "stateless_layernorm_per_call",
            "weight_tying": "each_stage_sequence_shared_across_its_repeats",
            "stochastic_depth_prob": 0.0,
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
        saved_depths = tuple(checkpoint.get("stage_depths", ()))
        saved_repeats = tuple(checkpoint.get("stage_repeats", ()))
        if saved_depths != stage_depths or saved_repeats != stage_repeats:
            raise ValueError(
                "Resume architecture mismatch: "
                f"checkpoint ARR1/ARR2={saved_depths}/{saved_repeats}, "
                f"current={stage_depths}/{stage_repeats}"
            )
        saved_convnext_version = validate_convnext_version(
            checkpoint.get(
                "convnext_version", saved.get("convnext_version", 1)
            )
        )
        if saved_convnext_version != convnext_version:
            raise ValueError(
                "Resume architecture mismatch: "
                f"checkpoint V={saved_convnext_version}, current V={convnext_version}"
            )
        saved_reg_mode, saved_n_reg = validate_register_arrays(
            checkpoint.get("reg_mode", saved.get("reg_mode", DEFAULT_REG_MODE)),
            checkpoint.get("n_reg", saved.get("n_reg", DEFAULT_N_REG)),
            saved_depths,
        )
        if saved_reg_mode != reg_mode or saved_n_reg != n_reg:
            raise ValueError(
                "Resume register architecture mismatch: "
                f"checkpoint REG_MODE/N_REG={saved_reg_mode}/{saved_n_reg}, "
                f"current={reg_mode}/{n_reg}"
            )
        saved_delta_mode = bool(
            checkpoint.get("delta_mode", saved.get("delta_mode", False))
        )
        saved_reg_head = bool(
            checkpoint.get("reg_head", saved.get("reg_head", False))
        )
        if saved_delta_mode != delta_mode or saved_reg_head != reg_head:
            raise ValueError(
                "Resume register architecture mismatch: "
                f"checkpoint DELTA_MODE/REG_HEAD={saved_delta_mode}/{saved_reg_head}, "
                f"current={delta_mode}/{reg_head}"
            )
        critical = (
            "model_family", "activation_checkpointing", "img", "resize", "epochs",
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
            "format_version": 4,
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
            "convnext_version": convnext_version,
            "reg_mode": reg_mode,
            "n_reg": n_reg,
            "delta_mode": delta_mode,
            "reg_head": reg_head,
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
            stage_depths,
            stage_repeats,
            num_classes,
            convnext_version=convnext_version,
            reg_mode=reg_mode,
            n_reg=n_reg,
            delta_mode=delta_mode,
            reg_head=reg_head,
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
            f"memory_probe model=convnext V={convnext_version} batch_size={targets.numel()} "
            f"peak_allocated_gib={torch.cuda.max_memory_allocated(device) / 1024**3:.3f} "
            f"peak_reserved_gib={torch.cuda.max_memory_reserved(device) / 1024**3:.3f}"
        )

    def train(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = RecurrentCNN(
            stage_depths,
            stage_repeats,
            num_classes,
            convnext_version=convnext_version,
            reg_mode=reg_mode,
            n_reg=n_reg,
            delta_mode=delta_mode,
            reg_head=reg_head,
        ).to(device)
        params = sum(parameter.numel() for parameter in model.parameters())
        run_dir = os.path.join(output_dir, f"convnext_seed{seed}")
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
                    desc=f"convnext s{seed} ep {epoch + 1}/{epochs}",
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
                f"[convnext s{seed}] ep{epoch:2d} train_loss={train_loss:.3f} "
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
            "residual_by_stage": residuals,
        }

    log(f"data={data_root} classes={num_classes} train_batches={len(train_loader)} val_batches={len(val_loader)}")
    log(f"device={device} distributed={distributed} world_size={world_size} amp={amp_enabled} amp_dtype={amp_dtype_name}")
    log(
        f"model=convnext V={convnext_version} ARR1={arr1_slug} ARR2={arr2_slug} "
        f"REG_MODE={reg_mode_slug} N_REG={n_reg_slug} "
        f"DELTA_MODE={int(delta_mode)} REG_HEAD={int(reg_head)} "
        f"active_stages={active_stages} last_width={last_width} "
        f"unique_blocks={unique_blocks} block_applications={block_applications} "
        f"register_stages={register_stage_count} register_applications={register_applications} "
        "normalization_state=stateless_layernorm_per_call "
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
            summary_path = os.path.join(output_dir, "convnext_summary.json")
            save_json(summary_path, {
                "config": config(seeds[0], results[0]["parameters"]),
                "runs": results,
            })
            log(f"summary={summary_path}")
            for result in results:
                log(
                    f"seed={result['seed']} params={result['parameters']:,} "
                    f"best_val1={result['best_val1']:.6f} "
                    f"residual_by_stage={result['residual_by_stage']}"
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
