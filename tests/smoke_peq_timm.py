"""Run a minimal CPU pretrain-to-finetune checkpoint smoke test."""

import tempfile
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from peq_timm_common import create_stage_parser, run_stage


def run(stage, *arguments):
    args = create_stage_parser(stage).parse_args([*map(str, arguments)])
    run_stage(args, stage)


def make_dataset(root):
    for split in ("train", "val"):
        for class_index in range(2):
            directory = root / split / str(class_index)
            directory.mkdir(parents=True)
            for image_index in range(1):
                value = 32 + 96 * class_index + image_index
                Image.new("RGB", (12, 12), (value, value, value)).save(
                    directory / f"{image_index}.png"
                )


def common(data_root):
    return (
        "--data-root", data_root,
        "--device", "cpu",
        "--no-amp",
        "--delta-backend", "naive",
        "--sdpa-backend", "auto",
        "--wandb-mode", "disabled",
        "--workers", "0",
        "--image-size", "8",
        "--patch-size", "4",
        "--dim", "12",
        "--heads", "3",
        "--steps", "1",
        "--n-reg", "2",
        "--batch-size", "2",
        "--validation-batch-size", "2",
        "--mixup", "0",
        "--cutmix", "0",
    )


def main():
    with tempfile.TemporaryDirectory(prefix="peq-timm-smoke-") as temporary:
        root = Path(temporary)
        data_root = root / "data"
        make_dataset(data_root)
        pretrain_root = root / "pretrain"
        run(
            "pretrain",
            *common(data_root),
            "--output-dir", pretrain_root,
            "--epochs", "1",
            "--save-every", "1",
            "--opt", "adamw",
            "--lr", "1e-3",
            "--warmup-epochs", "0",
        )
        checkpoints = list(pretrain_root.parent.glob(
            pretrain_root.name + "_*/checkpoint_final.pt"
        ))
        if len(checkpoints) != 1:
            raise RuntimeError(f"Expected one pretrain checkpoint, found {checkpoints}")
        checkpoint = checkpoints[0]
        run(
            "finetune",
            *common(data_root),
            "--output-dir", root / "finetune",
            "--pretrained-checkpoint", checkpoint,
            "--epochs", "1",
            "--opt", "adamw",
            "--lr", "1e-4",
            "--warmup-epochs", "0",
        )


if __name__ == "__main__":
    main()
