"""
Patch-DeltaNet PEQ on ImageNet-1K with the DATA TERM and multi-seed error bars.

Each block first applies a DeltaNet residual, followed by the PEQ register stages.
By default (DELTAREG=0), DeltaNet processes patches only. With DELTAREG=1, all
registers active at the current iteration are appended after the patches and the
joint sequence is processed by the same DeltaNet parameters. ATTN=sequential
uses independent softmax attention in each register stage. ATTN=rats shares
Wq/Wk/Wv across the three stages and bypasses register projections in refine and
broadcast.

  x = x + patch_delta(norm(x))                         # DELTAREG=0
  [x, r] = [x, r] + patch_delta(norm([x, r]))          # DELTAREG=1
  r = r + compress(norm(r), norm(x), norm(x))
  r = r + refine(norm(r))
  x = x + broadcast(norm(x), norm(r), norm(r))
  x = x + mlp(norm(x))

Motivation (2026-06-27 external reviews + PEQ Theoretical Foundations):
the old refine step was internal-consensus only -- the image entered just via
compress, so the "equilibrium" never explained the image. Here the registers
PREDICT the tokens and the prediction error drives their update (predictive
coding): the fixed point now minimizes E_data(X,R)+E_prior(R), not E_prior alone.

Modes (all parameter-matched; data term reuses compress/broadcast weights, adds
only one scalar gamma_d):
  single        : 1 block, T=1                         (floor)
  tied          : tied block x T, consensus refine     (prior reproduction)
  untied        : T distinct blocks, consensus refine  (depth reference, rung 3)
  tied_data     : tied block x T, refine + DATA TERM   (the new improvement)
  tied_data_rec : tied_data + aux reconstruction loss  (equilibrium trained to explain image)

Diagnostics (logged for every mode, seed 0):
  resid/step   ||dr||/||r||  -> converge (PEQ) vs unroll (bViT)
  recon/step   mean(eps^2)   -> data energy; should DESCEND across T only for *_data
Multi-seed (SEEDS) gives the owed rung-3 error bars: tied vs untied, data vs no-data.
"""
from contextlib import nullcontext

import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.parallel import DistributedDataParallel as DDP

from deltanet import DELTA_BACKENDS, DeltaNet, require_fla
from imagenet_data import make_imagenet_loaders

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def env_flag(name, default=0):
    return bool(int(os.environ.get(name, default)))


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if env_flag("REQUIRE_CUDA", 0) and not torch.cuda.is_available():
        raise RuntimeError(
            "REQUIRE_CUDA=1 but torch.cuda.is_available() is False. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            f"cuda_device_count={torch.cuda.device_count()}. "
            "Check the Slurm GPU allocation, node health, and the Python/PyTorch CUDA environment."
        )
    distributed = world_size > 1
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
    return distributed, rank, local_rank, world_size


def is_main():
    return rank == 0


def log(*args, **kwargs):
    if is_main():
        print(*args, **kwargs)


def progress_bar(iterable, desc):
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        disable=(not PROGRESS) or (not is_main()),
        dynamic_ncols=True,
        leave=False,
    )


distributed, rank, local_rank, world_size = setup_distributed()
dev = f"cuda:{local_rank}" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
device_type = "cuda" if str(dev).startswith("cuda") else ("mps" if dev == "mps" else "cpu")

RESUME = os.environ.get("RESUME", "").strip()
resume_checkpoint = None
if RESUME:
    if not os.path.isfile(RESUME):
        raise FileNotFoundError(f"Resume checkpoint not found: {RESUME}")
    resume_checkpoint = torch.load(RESUME, map_location="cpu", weights_only=False)
    resume_config = resume_checkpoint.get("config")
    if not isinstance(resume_config, dict):
        raise ValueError(f"Resume checkpoint has no valid config: {RESUME}")
    if resume_config.get("model_family") != "delta_peq":
        raise ValueError(
            f"Resume checkpoint is not a delta_peq checkpoint: {RESUME}"
        )
    if resume_config.get("model_arch") not in {
        "sequential_patch_delta_softmax_stages_v1",
        "rats_patch_delta_shared_qkv_identity_registers_v1",
        "sequential_patch_register_delta_softmax_stages_v1",
        "rats_patch_register_delta_shared_qkv_identity_registers_v1",
    }:
        raise ValueError(
            "Only sequential and RATS Delta-PEQ checkpoints are supported: "
            f"{RESUME}"
        )
    env_from_config = {
        "D": "dim",
        "N_REG": "n_reg",
        "ATTN": "attn",
        "SDPA_BACKEND": "sdpa_backend",
        "DELTA_BACKEND": "delta_backend",
        "DELTA_CHUNK_SIZE": "delta_chunk_size",
        "DELTAREG": "deltareg",
        "READOUT": "readout",
        "MIDOUT": "midout",
        "RMSNORM": "rmsnorm",
        "LAYERSCALE": "layerscale",
        "LS_INIT": "ls_init",
        "DATA_ROOT": "data_root",
        "IMG": "img",
        "RESIZE": "resize",
        "EPOCHS": "epochs",
        "BS": "batch_size",
        "WORKERS": "workers",
        "T": "T",
        "GAMMA_D": "gamma_d0",
        "LREC": "lrec",
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
        value = resume_config.get(config_name)
        if value is not None:
            if isinstance(value, bool):
                value = int(value)
            elif isinstance(value, (list, tuple)):
                value = ",".join(str(item) for item in value)
            os.environ[env_name] = str(value)

AMP = env_flag("AMP", 0)
PROGRESS = env_flag("PROGRESS", 1)
MEMORY_PROBE = env_flag("MEMORY_PROBE", 0)
AMP_DTYPE = os.environ.get("AMP_DTYPE", "bfloat16").lower()
AMP_DTYPES = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16, "fp16": torch.float16}
amp_dtype = AMP_DTYPES.get(AMP_DTYPE)
if AMP and amp_dtype is None:
    raise ValueError(f"Unsupported AMP_DTYPE={AMP_DTYPE}; use bfloat16 or float16")
use_amp = AMP and device_type in {"cuda", "cpu"}
scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16 and device_type == "cuda")


def autocast_ctx():
    if use_amp:
        return torch.autocast(device_type=device_type, dtype=amp_dtype)
    return nullcontext()

D = int(os.environ.get("D", 384))
T = int(os.environ.get("T", 4))
if T < 1:
    raise ValueError(f"T={T} must be positive")
N_REG_RAW = os.environ.get("N_REG", "8").strip()
if N_REG_RAW.startswith("[") and N_REG_RAW.endswith("]"):
    N_REG_RAW = N_REG_RAW[1:-1]
N_REG_PARTS = [part.strip() for part in N_REG_RAW.split(",")]
if not N_REG_PARTS or any(not part for part in N_REG_PARTS):
    raise ValueError("N_REG must be a positive integer or a comma-separated list of length T")
try:
    N_REG_VALUES = [int(part) for part in N_REG_PARTS]
except ValueError as exc:
    raise ValueError(
        "N_REG must be a positive integer or a comma-separated list of length T"
    ) from exc
if any(value < 1 for value in N_REG_VALUES):
    raise ValueError(f"N_REG values must be positive, got {N_REG_VALUES}")
if len(N_REG_VALUES) == 1:
    N_REG = N_REG_VALUES[0]
    N_REG_SCHEDULE = N_REG_VALUES * T
elif len(N_REG_VALUES) == T:
    N_REG = N_REG_VALUES
    N_REG_SCHEDULE = N_REG_VALUES
else:
    raise ValueError(
        f"N_REG must contain either 1 value or T={T} values, got {len(N_REG_VALUES)}"
    )
MAX_N_REG = max(N_REG_SCHEDULE)
N_REG_LABEL = "x".join(str(value) for value in N_REG_VALUES)
HEADS, PATCH = 6, 16
if D % HEADS != 0:
    raise ValueError(f"D={D} must be divisible by HEADS={HEADS}")
HD = D // HEADS
ATTN = os.environ.get("ATTN", "sequential").lower()
SDPA_BACKEND = os.environ.get("SDPA_BACKEND", "flash").lower()  # flash | auto
READOUT = os.environ.get("READOUT", "reg").lower()
MIDOUT = os.environ.get("MIDOUT", "none").lower()
MIDOUT_WEIGHT = 0.5
RMSNORM = env_flag("RMSNORM", 0)
LAYERSCALE = env_flag("LAYERSCALE", 0)
LS_INIT = float(os.environ.get("LS_INIT", 1e-4))
try:
    DELTAREG = int(os.environ.get("DELTAREG", 0))
except ValueError as exc:
    raise ValueError("DELTAREG must be 0 or 1") from exc
if DELTAREG not in {0, 1}:
    raise ValueError(f"DELTAREG={DELTAREG} must be 0 or 1")
DELTA_CONV_SIZE = 4
DELTA_NORM_EPS = 1e-5
DELTA_BACKEND = os.environ.get("DELTA_BACKEND", "auto").lower()
DELTA_CHUNK_SIZE = int(os.environ.get("DELTA_CHUNK_SIZE", 64))
if ATTN not in {"sequential", "rats"}:
    raise ValueError(f"Unsupported ATTN={ATTN}; use sequential or rats")
if READOUT not in {"reg", "weighted", "patch", "sum", "concat"}:
    raise ValueError(
        f"Unsupported READOUT={READOUT}; use reg, weighted, patch, sum, or concat"
    )
if MIDOUT not in {"none", "untied"}:
    raise ValueError(f"Unsupported MIDOUT={MIDOUT}; use none or untied")
if SDPA_BACKEND not in {"flash", "auto"}:
    raise ValueError(
        f"Unsupported SDPA_BACKEND={SDPA_BACKEND}; use flash or auto"
    )
if (
    device_type == "cuda"
    and SDPA_BACKEND == "flash"
    and not torch.backends.cuda.is_flash_attention_available()
):
    raise RuntimeError("SDPA_BACKEND=flash but PyTorch Flash SDPA is unavailable")
PATCH_ATTN = "standard_delta_patch_register" if DELTAREG else "standard_delta"
STAGE_LAYOUT = ATTN
if ATTN == "rats":
    COMPRESS_ATTN = "rats_shared_qkv"
    REFINE_ATTN = "rats_identity_registers"
    BROADCAST_ATTN = "rats_identity_register_kv"
else:
    COMPRESS_ATTN = "softmax"
    REFINE_ATTN = "softmax"
    BROADCAST_ATTN = "softmax"
DELTA_BACKEND_LABEL = DELTA_BACKEND
SDPA_BACKEND_LABEL = SDPA_BACKEND
if DELTAREG:
    MODEL_ARCH = (
        "rats_patch_register_delta_shared_qkv_identity_registers_v1"
        if ATTN == "rats"
        else "sequential_patch_register_delta_softmax_stages_v1"
    )
else:
    MODEL_ARCH = (
        "rats_patch_delta_shared_qkv_identity_registers_v1"
        if ATTN == "rats"
        else "sequential_patch_delta_softmax_stages_v1"
    )
if DELTA_BACKEND not in DELTA_BACKENDS:
    options = ", ".join(sorted(DELTA_BACKENDS))
    raise ValueError(f"Unsupported DELTA_BACKEND={DELTA_BACKEND}; use {options}")
if DELTA_CHUNK_SIZE not in {16, 32, 64}:
    raise ValueError("DELTA_CHUNK_SIZE must be 16, 32, or 64")
if DELTA_BACKEND in {"fla", "chunk"} and not use_amp:
    raise ValueError(f"DELTA_BACKEND={DELTA_BACKEND} requires AMP=1")
if DELTA_BACKEND in {"fla", "chunk", "fused_recurrent"}:
    require_fla()
DATA_ROOT = os.environ.get("DATA_ROOT", "/cis/home/cyang140/datasets/imagenet")
IMG = int(os.environ.get("IMG", 128))
RESIZE = int(os.environ.get("RESIZE", round(IMG * 146 / 128)))
L = (IMG // PATCH) ** 2
EPOCHS, BS = int(os.environ.get("EPOCHS", 22)), int(os.environ.get("BS", 256))
WORKERS = int(os.environ.get("WORKERS", 8))
GAMMA_D = float(os.environ.get("GAMMA_D", 0.5))   # init for the data-term step size
LREC = float(os.environ.get("LREC", 0.3))         # aux reconstruction loss weight
MODES = os.environ.get("MODES", "single,tied,untied,tied_data,tied_data_rec").split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
LIMIT_TRAIN = int(os.environ.get("LIMIT_TRAIN", 0))
LIMIT_VAL = int(os.environ.get("LIMIT_VAL", 0))
DELTAREG_OUTPUT_SUFFIX = "_dreg1" if DELTAREG else ""
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    f"outputs/imagenet_delta_peq_patchdelta{DELTAREG_OUTPUT_SUFFIX}_{STAGE_LAYOUT}_refine{REFINE_ATTN}_{DELTA_BACKEND_LABEL}_sdpa{SDPA_BACKEND_LABEL}_c{DELTA_CHUNK_SIZE}_D{D}_NREG{N_REG_LABEL}_T{T}_img{IMG}_readout{READOUT}_midout{MIDOUT}",
)
if resume_checkpoint is not None:
    saved_mode = resume_checkpoint.get("mode", resume_config.get("mode"))
    saved_seed = int(resume_checkpoint.get("seed", resume_config.get("seed", 0)))
    if saved_mode is None:
        raise ValueError("Resume checkpoint does not identify its training mode")
    MODES = [saved_mode]
    SEEDS = [saved_seed]
    OUTPUT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(RESUME)))
if int(os.environ.get("SMOKE", 0)):
    if RESUME:
        raise ValueError("SMOKE cannot be combined with RESUME")
    EPOCHS, SEEDS = 1, [0]
    LIMIT_TRAIN = LIMIT_TRAIN or 2048
    LIMIT_VAL = LIMIT_VAL or 1024

DATA_MODES = {"tied_data", "tied_data_rec"}
train_loader, val_loader, NUM_CLASSES, train_sampler = make_imagenet_loaders(
    DATA_ROOT,
    IMG,
    RESIZE,
    BS,
    WORKERS,
    LIMIT_TRAIN,
    LIMIT_VAL,
    distributed,
    rank,
    world_size,
    persistent_workers=False,
)
MAX_LR = float(os.environ.get("MAX_LR", 5e-4))
MIN_LR = float(os.environ.get("MIN_LR", 1e-6))
WARMUP_EPOCHS = int(os.environ.get("WARMUP_EPOCHS", 5))
if WARMUP_EPOCHS < 0:
    raise ValueError(f"WARMUP_EPOCHS={WARMUP_EPOCHS} must be non-negative")
GRAD_ACCUM_STEPS = int(os.environ.get("GRAD_ACCUM_STEPS", 1))
if GRAD_ACCUM_STEPS < 1:
    raise ValueError(f"GRAD_ACCUM_STEPS={GRAD_ACCUM_STEPS} must be positive")


def unwrap_model(net):
    return net.module if isinstance(net, DDP) else net


def expand_legacy_layerscale_state(net, state_dict):
    """Expand a legacy tied LayerScale vector into one row per iteration."""
    target_state = net.state_dict()
    migrated = state_dict.copy()
    expanded_names = set()
    added_numel = 0
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
        added_numel += target.numel() - source.numel()
    return migrated, expanded_names, added_numel


def expand_legacy_layerscale_optimizer(opt, net, expanded_names):
    """Match Adam moments to LayerScale vectors expanded during checkpoint load."""
    parameters = dict(unwrap_model(net).named_parameters())
    for name in expanded_names:
        parameter = parameters[name]
        for key, value in list(opt.state[parameter].items()):
            if (
                torch.is_tensor(value)
                and value.ndim + 1 == parameter.ndim
                and value.shape == parameter.shape[1:]
            ):
                opt.state[parameter][key] = (
                    value.unsqueeze(0).expand(parameter.shape).clone()
                )


def metric_sums(values):
    stats = torch.tensor(values, device=dev, dtype=torch.float64)
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return stats.tolist()


def run_config(mode, seed, nparam):
    return {
        "model_family": "delta_peq",
        "model_arch": MODEL_ARCH,
        "data_root": DATA_ROOT,
        "img": IMG,
        "resize": RESIZE,
        "patch": PATCH,
        "dim": D,
        "n_reg": N_REG,
        "n_reg_schedule": N_REG_SCHEDULE,
        "max_n_reg": MAX_N_REG,
        "heads": HEADS,
        "head_dim": HD,
        "attn": ATTN,
        "stage_layout": STAGE_LAYOUT,
        "sdpa_backend": SDPA_BACKEND,
        "patch_attention": PATCH_ATTN,
        "compress_attention": COMPRESS_ATTN,
        "refine_attention": REFINE_ATTN,
        "broadcast_attention": BROADCAST_ATTN,
        "delta_backend": DELTA_BACKEND,
        "delta_chunk_size": DELTA_CHUNK_SIZE,
        "deltareg": DELTAREG,
        "delta_conv_size": DELTA_CONV_SIZE,
        "delta_norm_eps": DELTA_NORM_EPS,
        "readout": READOUT,
        "midout": MIDOUT,
        "midout_weight": MIDOUT_WEIGHT if MIDOUT == "untied" else 0.0,
        "midout_iteration": max(1, round((1 if mode == "single" else T) / 2)),
        "rmsnorm": RMSNORM,
        "layerscale": LAYERSCALE,
        "layerscale_layout": "per_iteration" if LAYERSCALE else "disabled",
        "ls_init": LS_INIT,
        "T": T,
        "epochs": EPOCHS,
        "batch_size": BS,
        "workers": WORKERS,
        "limit_train": LIMIT_TRAIN,
        "limit_val": LIMIT_VAL,
        "gamma_d0": GAMMA_D,
        "lrec": LREC,
        "max_lr": MAX_LR,
        "min_lr": MIN_LR,
        "warmup_epochs": WARMUP_EPOCHS,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "optimizer": "adamw",
        "weight_decay": 0.05,
        "label_smoothing": 0.1,
        "amp": use_amp,
        "amp_dtype": AMP_DTYPE,
        "num_classes": NUM_CLASSES,
        "mode": mode,
        "seed": seed,
        "params_m": nparam,
    }


def save_history(history_path, history):
    if not is_main():
        return
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def save_curves(curve_path, epochs):
    if not is_main() or not epochs:
        return
    try:
        os.makedirs(os.path.join(OUTPUT_DIR, ".matplotlib"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, ".cache"), exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(OUTPUT_DIR, ".matplotlib"))
        os.environ.setdefault("XDG_CACHE_HOME", os.path.join(OUTPUT_DIR, ".cache"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"warning: could not save curves to {curve_path}: {exc}")
        return

    xs = [e["epoch"] for e in epochs]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(xs, [e["train_loss"] for e in epochs], label="train")
    axes[0, 0].plot(xs, [e["val_loss"] for e in epochs], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(xs, [e["train_acc1"] for e in epochs], label="train@1")
    axes[0, 1].plot(xs, [e["val_acc1"] for e in epochs], label="val@1")
    axes[0, 1].set_title("Top-1 Accuracy")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(xs, [e["train_acc5"] for e in epochs], label="train@5")
    axes[1, 0].plot(xs, [e["val_acc5"] for e in epochs], label="val@5")
    axes[1, 0].set_title("Top-5 Accuracy")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].legend()

    axes[1, 1].plot(xs, [e["lr"] for e in epochs], label="lr")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(curve_path), exist_ok=True)
    fig.savefig(curve_path, dpi=160)
    plt.close(fig)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(dev) if device_type == "cuda" else None,
    }


def collect_rng_states():
    local_state = capture_rng_state()
    if not distributed:
        return [local_state]
    gathered = [None] * world_size if is_main() else None
    dist.gather_object(local_state, gathered, dst=0)
    return gathered


def restore_rng_state(checkpoint):
    states = checkpoint.get("rng_by_rank")
    if states is None:
        log("warning: legacy checkpoint has no RNG state; stochastic streams cannot resume exactly")
        return
    saved_world_size = int(checkpoint.get("rng_world_size", len(states)))
    if saved_world_size != world_size or len(states) != world_size:
        raise ValueError(
            "Cannot restore per-rank RNG with a different world size: "
            f"checkpoint={saved_world_size}, current={world_size}"
        )
    state = states[rank]
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device_type == "cuda" and state.get("cuda") is not None:
        torch.cuda.set_rng_state(state["cuda"], device=dev)


def save_checkpoint(
    path,
    net,
    opt,
    sched,
    epoch,
    best,
    acc1,
    acc5,
    mode,
    seed,
    nparam,
    history,
    rng_by_rank,
):
    if not is_main():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "format_version": 2,
        "epoch": epoch,
        "global_step": sched.last_epoch,
        "best_val1": best,
        "val1": acc1,
        "val5": acc5,
        "mode": mode,
        "seed": seed,
        "params_m": nparam,
        "model": unwrap_model(net).state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "config": run_config(mode, seed, nparam),
        "history": history,
        "world_size": world_size,
        "steps_per_epoch": len(train_loader),
        "rng_world_size": world_size,
        "rng_by_rank": rng_by_rank,
    }
    temporary_path = f"{path}.tmp"
    torch.save(state, temporary_path)
    os.replace(temporary_path, path)


def validate_resume_checkpoint(checkpoint, mode, seed, nparam, expanded_layerscale_numel=0):
    if checkpoint.get("mode", mode) != mode:
        raise ValueError(f"Resume mode mismatch: checkpoint={checkpoint.get('mode')}, current={mode}")
    if int(checkpoint.get("seed", seed)) != seed:
        raise ValueError(f"Resume seed mismatch: checkpoint={checkpoint.get('seed')}, current={seed}")
    saved_world_size = checkpoint.get("world_size")
    if saved_world_size is not None and int(saved_world_size) != world_size:
        raise ValueError(
            f"Resume world size mismatch: checkpoint={saved_world_size}, current={world_size}"
        )
    saved_steps = checkpoint.get("steps_per_epoch")
    if saved_steps is not None and int(saved_steps) != len(train_loader):
        raise ValueError(
            "Resume train-loader length mismatch: "
            f"checkpoint={saved_steps}, current={len(train_loader)}"
        )
    saved_params = float(checkpoint.get("params_m", nparam))
    if not math.isclose(saved_params, nparam, rel_tol=0.0, abs_tol=1e-6):
        migrated_params = saved_params + expanded_layerscale_numel / 1e6
        if not math.isclose(migrated_params, nparam, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"Resume parameter count mismatch: checkpoint={saved_params}, current={nparam}")
    saved_config = checkpoint.get("config", {})
    current_config = run_config(mode, seed, nparam)
    critical_keys = (
        "model_family", "model_arch", "img", "resize", "patch", "dim", "n_reg",
        "n_reg_schedule", "max_n_reg", "heads",
        "attn", "stage_layout", "sdpa_backend", "patch_attention", "compress_attention", "refine_attention",
        "broadcast_attention", "delta_backend", "delta_chunk_size", "deltareg", "delta_conv_size",
        "delta_norm_eps", "readout", "midout", "midout_weight", "midout_iteration",
        "rmsnorm", "layerscale", "layerscale_layout",
        "ls_init", "T", "epochs",
        "batch_size", "workers", "limit_train", "limit_val", "gamma_d0", "lrec",
        "max_lr", "min_lr", "warmup_epochs", "grad_accum_steps", "optimizer", "weight_decay",
        "label_smoothing", "amp", "amp_dtype", "num_classes",
    )
    mismatches = [
        f"{key}: checkpoint={saved_config[key]!r}, current={current_config[key]!r}"
        for key in critical_keys
        if key in saved_config and saved_config[key] != current_config[key]
    ]
    if mismatches:
        raise ValueError("Resume configuration mismatch:\n  " + "\n  ".join(mismatches))


def resume_history(checkpoint, history_path, mode, seed, nparam):
    history = checkpoint.get("history")
    if not isinstance(history, dict) and os.path.isfile(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    if not isinstance(history, dict):
        history = {"config": run_config(mode, seed, nparam), "epochs": []}
    epochs = history.get("epochs")
    if not isinstance(epochs, list):
        epochs = []
    completed_epoch = int(checkpoint["epoch"])
    history["epochs"] = [record for record in epochs if int(record["epoch"]) <= completed_epoch]
    history["config"] = run_config(mode, seed, nparam)
    return history


def make_vit5_scheduler(opt, total_steps, steps_per_epoch):
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    min_factor = MIN_LR / MAX_LR
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return min_factor + (1.0 - min_factor) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xf.to(dtype) * self.weight.to(dtype)


def _norm(dim):
    return RMSNorm(dim, DELTA_NORM_EPS) if RMSNORM else nn.LayerNorm(dim)


def make_delta_attention():
    return DeltaNet(
        D,
        HEADS,
        DELTA_CONV_SIZE,
        DELTA_NORM_EPS,
        backend=DELTA_BACKEND,
        chunk_size=DELTA_CHUNK_SIZE,
    )


def softmax_attention(module, query, key, value, return_weights=False):
    context = (
        sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        if device_type == "cuda" and SDPA_BACKEND == "flash"
        else nullcontext()
    )
    with context:
        output, weights = module(
            query,
            key,
            value,
            need_weights=return_weights,
            average_attn_weights=True,
        )
    return (output, weights) if return_weights else output


class RATSAttention(nn.Module):
    """Three-stage attention with shared QKV and identity register projections."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.out_proj = nn.ModuleDict(
            {
                stage: nn.Linear(D, D)
                for stage in ("compress", "refine", "broadcast")
            }
        )

    @staticmethod
    def _split_heads(value):
        batch, length, _ = value.shape
        return value.reshape(batch, length, HEADS, HD).transpose(1, 2)

    def forward(self, stage, query, key, value, return_weights=False):
        if stage == "compress":
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)
        elif stage == "refine":
            # All three operands are registers and participate directly.
            pass
        elif stage == "broadcast":
            query = self.q_proj(query)
            # Register keys and values participate directly.
        else:
            raise ValueError(f"Unsupported RATS attention stage: {stage}")

        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        if return_weights:
            scores = query @ key.transpose(-2, -1) / math.sqrt(HD)
            head_weights = scores.softmax(dim=-1)
            context = head_weights @ value
            weights = head_weights.mean(dim=1)
        else:
            context_manager = (
                sdpa_kernel(SDPBackend.FLASH_ATTENTION)
                if device_type == "cuda" and SDPA_BACKEND == "flash"
                else nullcontext()
            )
            with context_manager:
                context = F.scaled_dot_product_attention(query, key, value)
            weights = None
        context = context.transpose(1, 2).contiguous().reshape(
            query.size(0), query.size(2), D
        )
        output = self.out_proj[stage](context)
        return (output, weights) if return_weights else output


class Block(nn.Module):
    def __init__(self, layerscale_steps=1):
        super().__init__()
        if layerscale_steps < 1:
            raise ValueError("layerscale_steps must be positive")
        self.lnp = _norm(D)
        self.patch_delta = make_delta_attention()
        self.lnm = _norm(D)
        self.w1 = nn.Linear(D, 4 * D)
        self.w2 = nn.Linear(4 * D, D)
        self.lnc = _norm(D)
        if ATTN == "rats":
            self.lnr = _norm(D)
            self.lnb = _norm(D)
            self.rats_attention = RATSAttention()
        else:
            self.compress = nn.MultiheadAttention(D, HEADS, batch_first=True)
            self.lnr = _norm(D)
            self.refine = nn.MultiheadAttention(D, HEADS, batch_first=True)
            self.lnb = _norm(D)
            self.broadcast = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.gamma_d = nn.Parameter(torch.tensor(GAMMA_D))
        if LAYERSCALE:
            # A tied block still needs an independently learnable scale at every
            # iteration. Keep the one-step shape for untied/single checkpoints.
            scale_shape = (D,) if layerscale_steps == 1 else (layerscale_steps, D)
            mk = lambda: nn.Parameter(LS_INIT * torch.ones(scale_shape))
            self.ls_p, self.ls_m = mk(), mk()
            self.ls_c, self.ls_rf, self.ls_d, self.ls_b = mk(), mk(), mk(), mk()

    def _scale(self, name, value, t):
        if not LAYERSCALE:
            return value
        scale = getattr(self, name)
        if scale.ndim == 2:
            scale = scale[t]
        return scale * value

    def forward(self, x, r, data, n_reg, t=0, return_broadcast_weights=False):
        if not 1 <= n_reg <= r.size(1):
            raise ValueError(
                f"Active register count must be in [1, {r.size(1)}], got {n_reg}"
            )
        inactive_r = r[:, n_reg:]
        r = r[:, :n_reg]
        if DELTAREG:
            # Registers follow patches so the causal DeltaNet treats the active
            # registers as additional tokens conditioned on the full patch
            # sequence. The same norm, DeltaNet, and LayerScale parameters are
            # shared by both token types; inactive registers remain untouched.
            delta_tokens = torch.cat((x, r), dim=1)
            delta_output = self._scale(
                "ls_p",
                self.patch_delta(self.lnp(delta_tokens)),
                t,
            )
            patch_output, register_output = delta_output.split(
                (x.size(1), n_reg), dim=1
            )
            x = x + patch_output
            r = r + register_output
        else:
            # Preserve the original patch-only DeltaNet path exactly.
            nx = self.lnp(x)
            patch_output = self.patch_delta(nx)
            x = x + self._scale("ls_p", patch_output, t)
        lx = self.lnc(x)
        compress_output = (
            self.rats_attention("compress", self.lnr(r), lx, lx)
            if ATTN == "rats"
            else softmax_attention(self.compress, self.lnr(r), lx, lx)
        )
        r = r + self._scale(
            "ls_c",
            compress_output,
            t,
        )                                                       # (i) recognition  L->N
        nr = self.lnr(r)
        refine_output = (
            self.rats_attention("refine", nr, nr, nr)
            if ATTN == "rats"
            else softmax_attention(self.refine, nr, nr, nr)
        )
        r = r + self._scale(
            "ls_rf",
            refine_output,
            t,
        )                                                       # (ii) prior/consensus  N->N
        bx = self.lnb(x)
        nr = self.lnr(r)
        broadcast_weights = None
        if return_broadcast_weights:
            if ATTN == "rats":
                xhat, attention_scores = self.rats_attention(
                    "broadcast", bx, nr, nr, return_weights=True
                )
            else:
                xhat, attention_scores = softmax_attention(
                    self.broadcast, bx, nr, nr, return_weights=True
                )
            # attention_scores: (batch, patch, reg). Aggregate how much all
            # patches attend to each register, then normalize over registers.
            broadcast_weights = attention_scores.sum(dim=1)
            broadcast_weights = broadcast_weights / broadcast_weights.sum(
                dim=-1, keepdim=True
            ).clamp_min(torch.finfo(broadcast_weights.dtype).eps)
        else:
            xhat = (
                self.rats_attention("broadcast", bx, nr, nr)
                if ATTN == "rats"
                else softmax_attention(self.broadcast, bx, nr, nr)
            )
        eps = bx - xhat                                          # prediction error (unexplained)
        if data:                                                 # (iii) DATA TERM: error-driven correction
            correction = (
                self.rats_attention("compress", self.lnr(r), eps, eps)
                if ATTN == "rats"
                else softmax_attention(self.compress, self.lnr(r), eps, eps)
            )
            r = r + self._scale(
                "ls_d",
                self.gamma_d * correction,
                t,
            )
        x = x + self._scale("ls_b", xhat, t)                     # broadcast write  N->L
        x = x + self._scale(
            "ls_m",
            self.w2(F.gelu(self.w1(self.lnm(x)))),
            t,
        )
        if inactive_r.size(1):
            r = torch.cat((r, inactive_r), dim=1)
        return x, r, eps.pow(2).mean(), broadcast_weights


class Net(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.data = mode in DATA_MODES
        self.T = 1 if mode == "single" else T
        self.n_reg_schedule = N_REG_SCHEDULE[:self.T]
        self.patch = nn.Conv2d(3, D, PATCH, PATCH)
        self.pos = nn.Parameter(torch.randn(1, L, D) * 0.02)
        self.r0 = nn.Parameter(torch.randn(1, MAX_N_REG, D) * 0.02)
        if mode == "untied":
            self.blocks = nn.ModuleList([Block() for _ in range(self.T)])
        else:
            self.block = Block(self.T)
        readout_dim = 2 * D if READOUT == "concat" else D
        self.head = nn.Sequential(_norm(readout_dim), nn.Linear(readout_dim, NUM_CLASSES))
        if MIDOUT == "untied":
            # The auxiliary classifier is intentionally independent from the
            # final classifier, including its normalization parameters.
            self.mid_head = nn.Sequential(
                _norm(readout_dim), nn.Linear(readout_dim, NUM_CLASSES)
            )
            # Iterations are counted from one in the MIDOUT definition.
            self.midout_iteration = max(1, round(self.T / 2))

    def _readout_features(self, x, r, n_reg, broadcast_weights):
        patch_mean = x.mean(1)
        active_r = r[:, :n_reg]
        if READOUT == "weighted":
            if broadcast_weights is None or broadcast_weights.shape != active_r.shape[:2]:
                raise RuntimeError(
                    "Weighted readout requires one broadcast weight per active register"
                )
            reg_features = (active_r * broadcast_weights.unsqueeze(-1)).sum(1)
        else:
            reg_features = active_r.mean(1)
        if READOUT in {"reg", "weighted"}:
            return reg_features
        if READOUT == "patch":
            return patch_mean
        if READOUT == "sum":
            return 0.5 * (reg_features + patch_mean)
        return torch.cat((reg_features, patch_mean), dim=-1)

    def forward(self, im, log=False, return_midout=False):
        if return_midout and MIDOUT != "untied":
            raise ValueError("return_midout=True requires MIDOUT=untied")
        x = self.patch(im).flatten(2).transpose(1, 2) + self.pos
        r = self.r0.expand(im.size(0), -1, -1).contiguous()
        resid, recon = [], []
        readout_weights = None
        mid_out = None
        for t in range(self.T):
            blk = self.blocks[t] if self.mode == "untied" else self.block
            n_reg = self.n_reg_schedule[t]
            r_prev = r[:, :n_reg]
            is_midout_iteration = (
                return_midout
                and MIDOUT == "untied"
                and t + 1 == self.midout_iteration
            )
            x, r, eps2, broadcast_weights = blk(
                x,
                r,
                self.data,
                n_reg,
                t,
                return_broadcast_weights=(
                    READOUT == "weighted"
                    and (t == self.T - 1 or is_midout_iteration)
                ),
            )
            if broadcast_weights is not None:
                readout_weights = broadcast_weights
            if is_midout_iteration:
                mid_features = self._readout_features(x, r, n_reg, broadcast_weights)
                mid_out = self.mid_head(mid_features)
            recon.append(eps2)
            if log:
                r_active = r[:, :n_reg]
                resid.append(
                    (r_active - r_prev).norm(dim=-1).mean().item()
                    / (r_active.norm(dim=-1).mean().item() + 1e-8)
                )
        final_n_reg = self.n_reg_schedule[-1]
        features = self._readout_features(x, r, final_n_reg, readout_weights)
        out = self.head(features)
        if return_midout:
            if mid_out is None:
                raise RuntimeError("MIDOUT iteration was not reached")
            return out, resid, recon, mid_out
        return out, resid, recon


def run_memory_probe():
    if device_type != "cuda":
        raise RuntimeError("MEMORY_PROBE=1 requires a CUDA device")
    if len(MODES) != 1:
        raise ValueError("MEMORY_PROBE=1 requires exactly one mode")

    mode = MODES[0]
    torch.manual_seed(SEEDS[0])
    net = Net(mode).to(dev).train()
    opt = torch.optim.AdamW(net.parameters(), lr=MAX_LR, weight_decay=0.05)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    xb, yb = next(iter(train_loader))
    xb = xb.to(dev, non_blocking=True)
    yb = yb.to(dev, non_blocking=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    with autocast_ctx():
        if MIDOUT == "untied":
            out, _, recon, mid_out = net(xb, return_midout=True)
        else:
            out, _, recon = net(xb)
            mid_out = None
        loss = lossf(out, yb)
        if mid_out is not None:
            loss = loss + MIDOUT_WEIGHT * lossf(mid_out, yb)
        if mode == "tied_data_rec":
            loss = loss + LREC * recon[-1]
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    torch.cuda.synchronize(dev)

    allocated = torch.cuda.max_memory_allocated(dev) / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved(dev) / (1024 ** 3)
    log(
        f"memory_probe mode={mode} batch_size={yb.numel()} "
        f"peak_allocated_gib={allocated:.3f} peak_reserved_gib={reserved:.3f}"
    )


def evaluate(net, lossf):
    net.eval(); loss_sum = 0.0; correct1 = 0; correct5 = 0; total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            with autocast_ctx():
                out = net(xb)[0]
                loss = lossf(out, yb)
            loss_sum += loss.detach().item() * yb.numel()
            correct1 += (out.argmax(1) == yb).sum().item()
            k = min(5, NUM_CLASSES)
            correct5 += out.topk(k, dim=1).indices.eq(yb[:, None]).any(1).sum().item()
            total += yb.numel()
    loss_sum, correct1, correct5, total = metric_sums([loss_sum, correct1, correct5, total])
    return loss_sum / total, correct1 / total, correct5 / total


def train(mode, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = Net(mode).to(dev)
    nparam = sum(p.numel() for p in net.parameters()) / 1e6
    ckpt_dir = os.path.join(OUTPUT_DIR, f"{mode}_seed{seed}")
    history_path = os.path.join(ckpt_dir, "history.json")
    curve_path = os.path.join(ckpt_dir, "curves.png")
    expanded_layerscale_names = set()
    if resume_checkpoint is not None:
        model_state, expanded_layerscale_names, added_numel = (
            expand_legacy_layerscale_state(net, resume_checkpoint["model"])
        )
        validate_resume_checkpoint(resume_checkpoint, mode, seed, nparam, added_numel)
        net.load_state_dict(model_state, strict=True)
        if expanded_layerscale_names:
            log(
                "migrated legacy tied LayerScale checkpoint to per-iteration "
                f"parameters: {sorted(expanded_layerscale_names)}"
            )
    if distributed:
        net = DDP(net, device_ids=[local_rank] if device_type == "cuda" else None, find_unused_parameters=True)
    opt = torch.optim.AdamW(net.parameters(), lr=MAX_LR, weight_decay=0.05)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM_STEPS)
    steps = EPOCHS * optimizer_steps_per_epoch
    sched = make_vit5_scheduler(opt, steps, optimizer_steps_per_epoch)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    start_epoch = 0
    best = -1.0
    final_acc1 = final_acc5 = 0.0
    history = {"config": run_config(mode, seed, nparam), "epochs": []}
    if resume_checkpoint is not None:
        opt.load_state_dict(resume_checkpoint["optimizer"])
        expand_legacy_layerscale_optimizer(opt, net, expanded_layerscale_names)
        sched.load_state_dict(resume_checkpoint["scheduler"])
        scaler_state = resume_checkpoint.get("scaler")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        best = float(resume_checkpoint.get("best_val1", -1.0))
        final_acc1 = float(resume_checkpoint.get("val1", 0.0))
        final_acc5 = float(resume_checkpoint.get("val5", 0.0))
        history = resume_history(resume_checkpoint, history_path, mode, seed, nparam)
        restore_rng_state(resume_checkpoint)
        log(
            f"resumed={RESUME} completed_epoch={start_epoch - 1} "
            f"next_epoch={start_epoch} global_step={sched.last_epoch} "
            f"lr={sched.get_last_lr()[0]:.6g} best_val1={best:.6f}"
        )
    elapsed_offset = history["epochs"][-1].get("elapsed_sec", 0.0) if history["epochs"] else 0.0
    t0 = time.time() - elapsed_offset
    for ep in range(start_epoch, EPOCHS):
        ep_t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        net.train()
        train_loss_sum = 0.0
        train_ce_sum = 0.0
        train_mid_ce_sum = 0.0
        train_correct1 = 0
        train_correct5 = 0
        train_total = 0
        pbar = progress_bar(train_loader, f"{mode} s{seed} ep {ep + 1}/{EPOCHS}")
        opt.zero_grad(set_to_none=True)
        for batch_idx, (xb, yb) in enumerate(pbar):
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            bsz = yb.numel()
            accumulation_start = (batch_idx // GRAD_ACCUM_STEPS) * GRAD_ACCUM_STEPS
            accumulation_size = min(GRAD_ACCUM_STEPS, len(train_loader) - accumulation_start)
            should_update = batch_idx + 1 == len(train_loader) or (batch_idx + 1) % GRAD_ACCUM_STEPS == 0
            sync_ctx = net.no_sync() if isinstance(net, DDP) and not should_update else nullcontext()
            with sync_ctx:
                with autocast_ctx():
                    if MIDOUT == "untied":
                        out, _, recon, mid_out = net(xb, return_midout=True)
                    else:
                        out, _, recon = net(xb)
                        mid_out = None
                    ce_loss = lossf(out, yb)
                    loss = ce_loss
                    if mid_out is not None:
                        mid_ce_loss = lossf(mid_out, yb)
                        loss = loss + MIDOUT_WEIGHT * mid_ce_loss
                    if mode == "tied_data_rec":
                        loss = loss + LREC * recon[-1]        # equilibrium trained to explain the image
                scaler.scale(loss / accumulation_size).backward()
            train_loss_sum += loss.detach().item() * bsz
            train_ce_sum += ce_loss.detach().item() * bsz
            if mid_out is not None:
                train_mid_ce_sum += mid_ce_loss.detach().item() * bsz
            train_correct1 += (out.argmax(1) == yb).sum().item()
            k = min(5, NUM_CLASSES)
            train_correct5 += out.topk(k, dim=1).indices.eq(yb[:, None]).any(1).sum().item()
            train_total += bsz
            if should_update:
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            if is_main() and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{loss.detach().item():.3f}", lr=f"{sched.get_last_lr()[0]:.2e}")
        train_loss_sum, train_ce_sum, train_mid_ce_sum, train_correct1, train_correct5, train_total = metric_sums(
            [train_loss_sum, train_ce_sum, train_mid_ce_sum, train_correct1, train_correct5, train_total]
        )
        train_loss = train_loss_sum / train_total
        train_ce_loss = train_ce_sum / train_total
        train_mid_ce_loss = (
            train_mid_ce_sum / train_total if MIDOUT == "untied" else None
        )
        train_acc1 = train_correct1 / train_total
        train_acc5 = train_correct5 / train_total
        val_loss, acc1, acc5 = evaluate(net, lossf)
        final_acc1, final_acc5 = acc1, acc5
        improved = acc1 > best
        best = max(best, acc1)
        epoch_record = {
            "epoch": ep,
            "train_loss": train_loss,
            "train_ce_loss": train_ce_loss,
            "train_mid_ce_loss": train_mid_ce_loss,
            "train_acc1": train_acc1,
            "train_acc5": train_acc5,
            "val_loss": val_loss,
            "val_acc1": acc1,
            "val_acc5": acc5,
            "best_val1": best,
            "lr": sched.get_last_lr()[0],
            "epoch_sec": time.time() - ep_t0,
            "elapsed_sec": time.time() - t0,
            "checkpoint_latest": os.path.join(ckpt_dir, "checkpoint_latest.pt"),
            "checkpoint_best": os.path.join(ckpt_dir, "checkpoint_best.pt") if improved else None,
            "checkpoint_final": (
                os.path.join(ckpt_dir, "checkpoint_final.pt") if ep + 1 == EPOCHS else None
            ),
        }
        history["epochs"].append(epoch_record)
        log(f"    [{mode:13s} s{seed}] ep{ep:2d} train_loss={train_loss:.3f} train@1={train_acc1:.3f} "
            f"val_loss={val_loss:.3f} val@1={acc1:.3f} val@5={acc5:.3f} best@1={best:.3f} ({time.time()-t0:.0f}s)")
        save_history(history_path, history)
        save_curves(curve_path, history["epochs"])
        rng_by_rank = collect_rng_states()
        save_checkpoint(
            os.path.join(ckpt_dir, "checkpoint_latest.pt"),
            net, opt, sched, ep, best, acc1, acc5, mode, seed, nparam,
            history, rng_by_rank,
        )
        if improved:
            save_checkpoint(
                os.path.join(ckpt_dir, "checkpoint_best.pt"),
                net, opt, sched, ep, best, acc1, acc5, mode, seed, nparam,
                history, rng_by_rank,
            )
        if ep + 1 == EPOCHS:
            save_checkpoint(
                os.path.join(ckpt_dir, "checkpoint_final.pt"),
                net, opt, sched, ep, best, acc1, acc5, mode, seed, nparam,
                history, rng_by_rank,
            )
    final_path = os.path.join(ckpt_dir, "checkpoint_final.pt")
    if start_epoch >= EPOCHS and not os.path.isfile(final_path):
        rng_by_rank = collect_rng_states()
        save_checkpoint(
            final_path,
            net, opt, sched, EPOCHS - 1, best, final_acc1, final_acc5, mode, seed, nparam,
            history, rng_by_rank,
        )
    if history["epochs"]:
        history["epochs"][-1]["checkpoint_final"] = final_path
        save_history(history_path, history)
        save_curves(curve_path, history["epochs"])
    net.eval()
    with torch.no_grad():
        xb, _ = next(iter(val_loader))
        with autocast_ctx():
            _, resid, recon = net(xb.to(dev, non_blocking=True), log=True)
    return nparam, best, resid, [r.item() for r in recon]


log(f"data={DATA_ROOT}  classes={NUM_CLASSES}  train_batches={len(train_loader)}  val_batches={len(val_loader)}")
log(f"output_dir={OUTPUT_DIR}")
log(f"device={dev}  distributed={distributed}  world_size={world_size}  amp={use_amp}  amp_dtype={AMP_DTYPE}")
log(
    f"img={IMG}  L={L} tok  N_schedule={N_REG_SCHEDULE} reg  "
    f"max_N={MAX_N_REG}  D={D}  heads={HEADS}  T={T}  "
    f"epochs={EPOCHS}  seeds={SEEDS}"
)
log(
    f"lr={MAX_LR:g}->{MIN_LR:g}  warmup_epochs={WARMUP_EPOCHS}  "
    f"grad_accum_steps={GRAD_ACCUM_STEPS}  effective_batch_size={BS * world_size * GRAD_ACCUM_STEPS}"
)
log(
    f"stage_layout={STAGE_LAYOUT}  patch_attn={PATCH_ATTN}  "
    f"compress_attn={COMPRESS_ATTN}  refine_attn={REFINE_ATTN}  "
    f"broadcast_attn={BROADCAST_ATTN}  readout={READOUT}  midout={MIDOUT}  "
    f"delta_backend={DELTA_BACKEND_LABEL}  sdpa_backend={SDPA_BACKEND_LABEL}  "
    f"delta_chunk_size={DELTA_CHUNK_SIZE}  deltareg={DELTAREG}  "
    f"delta_conv_size={DELTA_CONV_SIZE}  "
    f"delta_norm_eps={DELTA_NORM_EPS:g}"
)
log(f"rmsnorm={RMSNORM}  layerscale={LAYERSCALE}  ls_init={LS_INIT}")
log(f"modes={MODES}  gamma_d0={GAMMA_D}  lrec={LREC}\n")

if MEMORY_PROBE:
    run_memory_probe()
    if distributed:
        dist.destroy_process_group()
    raise SystemExit(0)

res = {}   # mode -> dict(params, accs[], resid, recon)
for mode in MODES:
    accs = []
    rd = rc = None
    for seed in SEEDS:
        npar, best, resid, recon = train(mode, seed)
        accs.append(best)
        if seed == SEEDS[0]:
            rd, rc = resid, recon
    import statistics as st
    mean = sum(accs) / len(accs)
    sd = st.pstdev(accs) if len(accs) > 1 else 0.0
    res[mode] = dict(params=npar, accs=accs, mean=mean, sd=sd, resid=rd, recon=rc)
    log(f"  -> {mode:13s}: params={npar:.2f}M  best_val={mean:.3f}+-{sd:.3f}  "
          f"accs={[round(a,3) for a in accs]}")
    log(f"     resid/step={[round(x,3) for x in (rd or [])]}  recon/step={[round(x,3) for x in (rc or [])]}\n")

log("=" * 72)
log(f"{'condition':14s} {'params(M)':>9s} {'val (mean+-sd)':>16s}   note")
def line(m, note=""):
    if m in res:
        r = res[m]
        log(f"{m:14s} {r['params']:>9.2f} {r['mean']:>9.3f}+-{r['sd']:.3f}   {note}")
line("single", "T=1 floor")
line("tied", "iteration @ fixed params")
line("untied", "depth reference (rung 3)")
line("tied_data", "+ DATA TERM (predictive coding)")
line("tied_data_rec", "+ aux reconstruction loss")

log("\n-- key contrasts (mean) --")
def gap(a, b):
    return res[a]['mean'] - res[b]['mean'] if a in res and b in res else float('nan')
if {"single", "tied"} <= set(res):       log(f"  tied  - single        = {gap('tied','single'):+.3f}   (iteration helps?)")
if {"tied", "untied"} <= set(res):       log(f"  tied  - untied        = {gap('tied','untied'):+.3f}   (rung 3: tying vs depth)")
if {"tied_data", "tied"} <= set(res):    log(f"  tied_data - tied      = {gap('tied_data','tied'):+.3f}   (does the DATA TERM help?)")
if {"tied_data_rec","tied_data"}<=set(res):log(f"  +rec - tied_data      = {gap('tied_data_rec','tied_data'):+.3f}   (does explaining the image help?)")

if distributed:
    dist.destroy_process_group()
