"""Sweep the stage-3 repeat count of an official recurrent ConvNeXt checkpoint.

The checkpoint's raw ``model`` weights and every architectural setting except
the third stage repeat count are held fixed. Validation is run for repeat
counts 1 through twice the value used during training.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, Subset

from timm.data import create_transform
from timm.utils import init_distributed_device

from deltanet import require_fla
from imagenet_data import NumericImageFolder
from recurrent_cnn import (
    CONV_MODEL_STAGE_WIDTHS,
    DELTA_BACKEND,
    DEFAULT_N_REG,
    DEFAULT_REG_MODE,
    RecurrentCNN,
    RepeatedStage,
    validate_convnext_version,
    validate_conv_model,
    validate_register_arrays,
    validate_stage_arrays,
)


LOG = logging.getLogger("recurrent_convnext_loop_eval")
CHECKPOINT_FORMAT_VERSION = 1
MODEL_FAMILY = "recurrent_cnn_official"
RESULT_FORMAT_VERSION = 1
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VALIDATION_CROP_PCT = 0.875
RESULT_JSON = "stage3_loop_sweep.json"
RESULT_CSV = "stage3_loop_sweep.csv"
RESULT_PNG = "stage3_loop_sweep.png"


@dataclass(frozen=True)
class CheckpointSpec:
    stage_depths: tuple[int, int, int, int]
    training_stage_repeats: tuple[int, int, int, int]
    convnext_version: int
    conv_model: str
    stage_widths: tuple[int, int, int, int]
    drop_path_rate: float
    reg_mode: tuple[int, int, int, int]
    n_reg: tuple[int, int, int, int]
    delta_mode: bool
    reg_head: bool
    num_classes: int
    image_size: int
    data_root: Optional[str]

    @property
    def training_stage3_repeats(self) -> int:
        return self.training_stage_repeats[2]

    @property
    def maximum_stage3_repeats(self) -> int:
        return 2 * self.training_stage3_repeats

    @property
    def maximum_stage_repeats(self) -> tuple[int, int, int, int]:
        repeats = list(self.training_stage_repeats)
        repeats[2] = self.maximum_stage3_repeats
        return tuple(repeats)  # type: ignore[return-value]


class DistributedEvalSampler(Sampler[int]):
    """Shard validation samples across ranks without padding or duplication."""

    def __init__(self, dataset, rank: int, world_size: int):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(f"Invalid distributed sampler rank/world_size={rank}/{world_size}")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self):
        return max(
            0,
            (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size,
        )


def parse_args(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate raw checkpoint accuracy while sweeping ConvNeXt stage-3 "
            "repeats from 1 to twice the training value"
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-root",
        default=None,
        help="ImageNet root containing val/; defaults to the checkpoint setting",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="defaults to CHECKPOINT_DIR/stage3_loop_sweep_raw",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="per-rank validation batch")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--limit-val",
        type=int,
        default=0,
        help="limit validation samples for tests; zero evaluates the complete set",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dist-backend", default=None)
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace this sweep's existing JSON, CSV, and PNG instead of resuming",
    )
    return parser.parse_args(argv)


def _required_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Checkpoint {name} must be a mapping")
    return value


def load_official_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} is not a dictionary")
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has incompatible format_version="
            f"{checkpoint.get('format_version')!r}; expected {CHECKPOINT_FORMAT_VERSION}"
        )
    if checkpoint.get("model_family") != MODEL_FAMILY:
        raise ValueError(
            f"Checkpoint {checkpoint_path} has model_family="
            f"{checkpoint.get('model_family')!r}; expected {MODEL_FAMILY!r}"
        )
    weights = checkpoint.get("model")
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("Checkpoint has no valid raw model state dictionary")
    return checkpoint


def checkpoint_spec(checkpoint: Mapping[str, Any]) -> CheckpointSpec:
    config = _required_mapping(checkpoint.get("config"), "config")
    architecture = _required_mapping(config.get("architecture"), "config.architecture")
    training = _required_mapping(config.get("training"), "config.training")
    dataset = _required_mapping(config.get("dataset"), "config.dataset")

    missing = [
        key
        for key in ("arr1", "arr2", "convnext_version")
        if key not in architecture
    ]
    if missing:
        raise ValueError(
            "Checkpoint architecture metadata is missing: " + ", ".join(missing)
        )
    depths, repeats = validate_stage_arrays(
        architecture["arr1"], architecture["arr2"]
    )
    if depths[2] == 0 or repeats[2] == 0:
        raise ValueError(
            "Stage 3 must be enabled in the checkpoint to run a stage-3 loop sweep"
        )
    reg_mode, n_reg = validate_register_arrays(
        architecture.get("reg_mode", DEFAULT_REG_MODE),
        architecture.get("n_reg", DEFAULT_N_REG),
        depths,
    )
    convnext_version = validate_convnext_version(architecture["convnext_version"])
    conv_model = validate_conv_model(
        architecture.get("conv_model", "t"), convnext_version, reg_mode
    )
    expected_widths = CONV_MODEL_STAGE_WIDTHS[conv_model]
    try:
        stage_widths = tuple(
            int(width)
            for width in architecture.get("stage_widths", expected_widths)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint stage_widths must contain four integers") from exc
    if stage_widths != expected_widths:
        raise ValueError(
            f"Checkpoint stage_widths={stage_widths} do not match "
            f"CONV_MODEL={conv_model}: expected {expected_widths}"
        )
    expected_last_width = expected_widths[sum(depth > 0 for depth in depths) - 1]
    try:
        last_width = int(architecture.get("last_width", expected_last_width))
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint last_width must be an integer") from exc
    if last_width != expected_last_width:
        raise ValueError("Checkpoint last_width does not match its active stages")

    weights = _required_mapping(checkpoint.get("model"), "raw model state dictionary")
    head_weight = weights.get("head.weight")
    inferred_classes = (
        int(head_weight.shape[0])
        if isinstance(head_weight, torch.Tensor) and head_weight.ndim == 2
        else None
    )
    configured_classes = dataset.get("num_classes", inferred_classes)
    if configured_classes is None:
        raise ValueError("Cannot determine the checkpoint class count")
    num_classes = int(configured_classes)
    if num_classes < 1:
        raise ValueError(f"Checkpoint num_classes must be positive, got {num_classes}")
    if inferred_classes is not None and inferred_classes != num_classes:
        raise ValueError(
            "Checkpoint dataset.num_classes does not match model head: "
            f"{num_classes} != {inferred_classes}"
        )

    image_size = int(training.get("image_size", 224))
    if image_size < 1:
        raise ValueError(f"Checkpoint image_size must be positive, got {image_size}")
    data_root = training.get("data_root")
    if data_root is not None and not isinstance(data_root, str):
        raise ValueError("Checkpoint training.data_root must be a string")

    return CheckpointSpec(
        stage_depths=depths,
        training_stage_repeats=repeats,
        convnext_version=convnext_version,
        conv_model=conv_model,
        stage_widths=stage_widths,
        drop_path_rate=float(architecture.get("drop_path_rate", 0.0)),
        reg_mode=reg_mode,
        n_reg=n_reg,
        delta_mode=bool(architecture.get("delta_mode", False)),
        reg_head=bool(architecture.get("reg_head", False)),
        num_classes=num_classes,
        image_size=image_size,
        data_root=data_root,
    )


def sweep_repeat_counts(training_repeats: int) -> list[int]:
    if training_repeats < 1:
        raise ValueError("The training stage-3 repeat count must be positive")
    return list(range(1, 2 * training_repeats + 1))


def repeated_stages(model: RecurrentCNN) -> list[RepeatedStage]:
    return [module for module in model.features if isinstance(module, RepeatedStage)]


def set_stage3_repeats(model: RecurrentCNN, repeats: int) -> None:
    stages = repeated_stages(model)
    if len(stages) < 3:
        raise ValueError("Model does not contain an active stage 3")
    maximum = len(stages[2].register_drop_path_probs)
    if not 1 <= repeats <= maximum:
        raise ValueError(f"Stage-3 repeats must be in [1, {maximum}], got {repeats}")
    stages[2].repeats = repeats


def current_stage_repeats(model: RecurrentCNN) -> tuple[int, ...]:
    return tuple(stage.repeats for stage in repeated_stages(model))


def build_sweep_model(spec: CheckpointSpec, weights: Mapping[str, Any]) -> RecurrentCNN:
    model = RecurrentCNN(
        spec.stage_depths,
        spec.maximum_stage_repeats,
        num_classes=spec.num_classes,
        convnext_version=spec.convnext_version,
        drop_path_rate=spec.drop_path_rate,
        reg_mode=spec.reg_mode,
        n_reg=spec.n_reg,
        delta_mode=spec.delta_mode,
        reg_head=spec.reg_head,
        conv_model=spec.conv_model,
    )
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Raw checkpoint weights do not match the saved architecture: {exc}") from exc
    return model


def create_validation_loader(args, spec: CheckpointSpec):
    data_root_value = args.data_root or spec.data_root
    if not data_root_value:
        raise ValueError("--data-root is required because the checkpoint has no data_root")
    data_root = Path(data_root_value).expanduser().resolve()
    val_dir = data_root / "val"
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Expected ImageNet validation directory: {val_dir}")

    transform = create_transform(
        input_size=(3, spec.image_size, spec.image_size),
        is_training=False,
        interpolation="bicubic",
        crop_pct=VALIDATION_CROP_PCT,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        use_prefetcher=False,
    )
    full_dataset = NumericImageFolder(val_dir, transform=transform)
    dataset_classes = max(full_dataset.class_to_idx.values()) + 1
    if dataset_classes != spec.num_classes:
        raise ValueError(
            f"Validation dataset has {dataset_classes} classes, checkpoint expects "
            f"{spec.num_classes}"
        )
    dataset = (
        Subset(full_dataset, range(min(args.limit_val, len(full_dataset))))
        if args.limit_val > 0
        else full_dataset
    )
    if len(dataset) == 0:
        raise ValueError("Validation dataset is empty")
    sampler = (
        DistributedEvalSampler(dataset, args.rank, args.world_size)
        if args.distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=args.device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    return loader, data_root, len(dataset)


def _autocast(args):
    if not args.amp:
        return nullcontext()
    dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type=args.device.type, dtype=dtype)


def _reduce_sums(values: list[float], device: torch.device) -> list[float]:
    totals = torch.tensor(values, device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return totals.tolist()


@torch.inference_mode()
def evaluate_model(args, model: nn.Module, loader: DataLoader) -> dict[str, float | int]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = correct1 = correct5 = samples = 0.0
    started = time.monotonic()
    for images, targets in loader:
        images = images.to(args.device, non_blocking=True)
        targets = targets.to(args.device, non_blocking=True)
        with _autocast(args):
            logits, _ = model(images)
            loss = criterion(logits, targets)
        loss_sum += loss.item()
        correct1 += logits.argmax(1).eq(targets).sum().item()
        correct5 += (
            logits.topk(min(5, logits.shape[1]), dim=1)
            .indices.eq(targets[:, None])
            .any(1)
            .sum()
            .item()
        )
        samples += targets.numel()
    loss_sum, correct1, correct5, samples = _reduce_sums(
        [loss_sum, correct1, correct5, samples], args.device
    )
    if samples <= 0:
        raise RuntimeError("No validation samples were evaluated")
    elapsed = torch.tensor(time.monotonic() - started, device=args.device)
    if dist.is_initialized():
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return {
        "val_loss": loss_sum / samples,
        "val_acc1": 100.0 * correct1 / samples,
        "val_acc5": 100.0 * correct5 / samples,
        "samples": int(samples),
        "elapsed_sec": float(elapsed.item()),
    }


def checkpoint_identity(path: Path, checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "epoch": int(checkpoint.get("epoch", -1)),
        "num_updates": int(checkpoint.get("num_updates", -1)),
        "weight_key": "model",
    }


def result_metadata(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    spec: CheckpointSpec,
    data_root: Path,
    val_size: int,
    args,
) -> dict[str, Any]:
    spec_dict = asdict(spec)
    spec_dict.pop("data_root")
    for key in (
        "stage_depths",
        "training_stage_repeats",
        "stage_widths",
        "reg_mode",
        "n_reg",
    ):
        spec_dict[key] = list(spec_dict[key])
    return {
        "format_version": RESULT_FORMAT_VERSION,
        "model_family": MODEL_FAMILY,
        "checkpoint": checkpoint_identity(checkpoint_path, checkpoint),
        "architecture": spec_dict,
        "sweep": {
            "stage": 3,
            "minimum_repeats": 1,
            "training_repeats": spec.training_stage3_repeats,
            "maximum_repeats": spec.maximum_stage3_repeats,
            "repeat_counts": sweep_repeat_counts(spec.training_stage3_repeats),
            "fixed_stage_repeats": list(spec.training_stage_repeats),
        },
        "dataset": {
            "data_root": str(data_root),
            "val_size": val_size,
            "image_size": spec.image_size,
            "crop_pct": VALIDATION_CROP_PCT,
        },
        "evaluation": {
            "weights": "raw",
            "amp": args.amp,
            "amp_dtype": args.amp_dtype if args.amp else None,
        },
    }


def atomic_json_save(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_csv_save(records: list[dict[str, Any]], path: Path) -> None:
    fieldnames = (
        "stage3_repeats",
        "val_acc1",
        "val_acc5",
        "val_loss",
        "samples",
        "elapsed_sec",
    )
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_plot(records: list[dict[str, Any]], training_repeats: int, path: Path) -> None:
    if not records:
        path.unlink(missing_ok=True)
        return
    matplotlib_dir = path.parent / ".matplotlib"
    cache_dir = path.parent / ".cache"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    ordered = sorted(records, key=lambda record: int(record["stage3_repeats"]))
    xs = [int(record["stage3_repeats"]) for record in ordered]
    ys = [float(record["val_acc1"]) for record in ordered]
    best_index = max(range(len(ys)), key=ys.__getitem__)

    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(xs, ys, color="#2563eb", marker="o", linewidth=1.8, label="Raw Val Top-1")
    axis.axvline(
        training_repeats,
        color="#f59e0b",
        linestyle="--",
        linewidth=1.5,
        label=f"Training repeats (T={training_repeats})",
    )
    axis.scatter(
        [xs[best_index]],
        [ys[best_index]],
        color="#dc2626",
        zorder=3,
        label=f"Best {ys[best_index]:.3f}% at {xs[best_index]}",
    )
    axis.annotate(
        f"{ys[best_index]:.3f}%",
        (xs[best_index], ys[best_index]),
        xytext=(7, 7),
        textcoords="offset points",
    )
    axis.set_title("ConvNeXt Stage 3 Loop Sweep (Raw Weights)")
    axis.set_xlabel("Stage 3 repeats")
    axis.set_ylabel("Validation Top-1 accuracy (%)")
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    temporary = path.with_name(path.name + ".tmp")
    figure.savefig(temporary, format="png", dpi=160)
    plt.close(figure)
    os.replace(temporary, path)


def save_artifacts(
    metadata: Mapping[str, Any], records: list[dict[str, Any]], output_dir: Path
) -> None:
    ordered = sorted(records, key=lambda record: int(record["stage3_repeats"]))
    payload = dict(metadata)
    payload["results"] = ordered
    atomic_json_save(payload, output_dir / RESULT_JSON)
    atomic_csv_save(ordered, output_dir / RESULT_CSV)
    save_plot(
        ordered,
        int(metadata["sweep"]["training_repeats"]),
        output_dir / RESULT_PNG,
    )


def load_existing_results(
    output_dir: Path, expected_metadata: Mapping[str, Any], overwrite: bool
) -> list[dict[str, Any]]:
    paths = [output_dir / RESULT_JSON, output_dir / RESULT_CSV, output_dir / RESULT_PNG]
    if overwrite:
        for path in paths:
            path.unlink(missing_ok=True)
        return []
    json_path = output_dir / RESULT_JSON
    if not json_path.exists():
        leftovers = [str(path) for path in paths[1:] if path.exists()]
        if leftovers:
            raise ValueError(
                "Found sweep artifacts without the authoritative JSON file; "
                "use --overwrite: " + ", ".join(leftovers)
            )
        return []
    with open(json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Existing result file is not a JSON object: {json_path}")
    results = payload.pop("results", None)
    if payload != dict(expected_metadata):
        raise ValueError(
            "Existing stage-3 sweep metadata does not match this run; use "
            "--overwrite or select another --output-dir"
        )
    if not isinstance(results, list):
        raise ValueError("Existing result JSON has no valid results list")
    allowed = set(expected_metadata["sweep"]["repeat_counts"])
    seen: set[int] = set()
    for record in results:
        if not isinstance(record, dict) or "stage3_repeats" not in record:
            raise ValueError("Existing result JSON contains a malformed record")
        repeats = int(record["stage3_repeats"])
        if repeats not in allowed or repeats in seen:
            raise ValueError(
                f"Existing result JSON has invalid or duplicate stage3_repeats={repeats}"
            )
        seen.add(repeats)
    return results


def _is_primary(args) -> bool:
    return args.rank == 0


def run(args) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.workers < 0 or args.limit_val < 0:
        raise ValueError("--workers and --limit-val must be non-negative")

    args.device = init_distributed_device(args)
    if args.device.type != "cuda" and args.amp:
        if _is_primary(args):
            LOG.warning("Disabling AMP on non-CUDA device %s", args.device)
        args.amp = False
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_official_checkpoint(checkpoint_path)
    spec = checkpoint_spec(checkpoint)
    loader, data_root, val_size = create_validation_loader(args, spec)
    weights = checkpoint["model"]

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else checkpoint_path.expanduser().resolve().parent / "stage3_loop_sweep_raw"
    )
    metadata = result_metadata(
        checkpoint_path, checkpoint, spec, data_root, val_size, args
    )
    del checkpoint
    if _is_primary(args):
        output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    if _is_primary(args):
        records = load_existing_results(output_dir, metadata, args.overwrite)
        save_artifacts(metadata, records, output_dir)
    if dist.is_initialized():
        dist.barrier()
    if not _is_primary(args):
        records = load_existing_results(output_dir, metadata, overwrite=False)

    completed = {int(record["stage3_repeats"]) for record in records}
    pending = [
        repeats
        for repeats in sweep_repeat_counts(spec.training_stage3_repeats)
        if repeats not in completed
    ]
    if not pending:
        if _is_primary(args):
            best = max(records, key=lambda record: float(record["val_acc1"]))
            LOG.info(
                "Sweep already complete: best val_acc1=%.3f at "
                "stage3_repeats=%d; output=%s",
                best["val_acc1"],
                best["stage3_repeats"],
                output_dir,
            )
        return

    if spec.delta_mode and DELTA_BACKEND != "naive":
        if args.device.type != "cuda":
            raise RuntimeError(
                f"DELTA_BACKEND={DELTA_BACKEND} requires CUDA for this checkpoint"
            )
        require_fla()
    model = build_sweep_model(spec, weights).to(args.device)
    model.eval()
    del weights
    expected_other_repeats = spec.maximum_stage_repeats
    for repeats in pending:
        set_stage3_repeats(model, repeats)
        active_repeats = current_stage_repeats(model)
        for stage_index, expected in enumerate(expected_other_repeats):
            if spec.stage_depths[stage_index] == 0 or stage_index == 2:
                continue
            if active_repeats[stage_index] != expected:
                raise RuntimeError(
                    f"Stage {stage_index + 1} repeats changed unexpectedly: "
                    f"{active_repeats[stage_index]} != {expected}"
                )
        metrics = evaluate_model(args, model, loader)
        record = {"stage3_repeats": repeats, **metrics}
        records.append(record)
        if _is_primary(args):
            save_artifacts(metadata, records, output_dir)
            LOG.info(
                "stage3_repeats=%d val_acc1=%.3f val_acc5=%.3f "
                "val_loss=%.4f samples=%d sec=%.1f",
                repeats,
                metrics["val_acc1"],
                metrics["val_acc5"],
                metrics["val_loss"],
                metrics["samples"],
                metrics["elapsed_sec"],
            )
        if dist.is_initialized():
            dist.barrier()

    if _is_primary(args):
        best = max(records, key=lambda record: float(record["val_acc1"]))
        LOG.info(
            "Sweep complete: best val_acc1=%.3f at stage3_repeats=%d; output=%s",
            best["val_acc1"],
            best["stage3_repeats"],
            output_dir,
        )


def main(argv: Optional[list[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args(argv)
    try:
        run(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
