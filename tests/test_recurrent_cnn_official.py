import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from PIL import Image
from timm.loss import SoftTargetCrossEntropy

from imagenet_recurrent_cnn_official import (
    RESUME_ARGUMENT_KEYS,
    DistributedEvalSampler,
    _restore_resume_arguments,
    append_architecture_suffix,
    architecture_suffix,
    best_raw_acc1_from_metrics,
    build_config,
    create_optimizer,
    create_train_criterion,
    create_update_scheduler,
    default_output_dir,
    default_run_name,
    effective_batch_size,
    main,
    official_recipe_mismatches,
    parse_args,
    scaled_peak_lr,
)
from recurrent_cnn import ATTO_STAGE_WIDTHS, TINY_STAGE_WIDTHS, RecurrentCNN


class RecurrentCNNOfficialTest(unittest.TestCase):
    def test_best_raw_acc1_from_metrics_respects_resume_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics_path = Path(directory) / "metrics.jsonl"
            records = [
                {"epoch": 0, "raw": {"acc1": 60.0}},
                {"epoch": 1, "raw": {"acc1": 72.5}},
                {"epoch": 2, "raw": {"acc1": 70.0}},
            ]
            metrics_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            self.assertEqual(best_raw_acc1_from_metrics(metrics_path), 72.5)
            self.assertEqual(
                best_raw_acc1_from_metrics(metrics_path, before_epoch=1),
                60.0,
            )

    def test_default_arguments_match_official_recipe(self):
        with patch.dict(os.environ, {"EPOCHS": "300", "WARMUP_EPOCHS": "20"}):
            args = parse_args([])
        args.world_size = 1
        self.assertEqual(effective_batch_size(args), 4096)
        self.assertEqual(scaled_peak_lr(args), 4e-3)
        self.assertEqual(official_recipe_mismatches(args), [])
        self.assertEqual(args.convnext_version, 2)
        self.assertEqual(args.conv_model, "t")
        self.assertEqual(args.arr1, "1,1,2,0")
        self.assertEqual(args.arr2, "3,3,6,0")
        self.assertEqual(args.reg_mode, "0,0,0,0")
        self.assertEqual(args.n_reg, "8,8,8,8")
        self.assertFalse(args.delta_mode)
        self.assertFalse(args.reg_head)

    def test_architecture_arguments_and_destination_suffixes(self):
        args = parse_args(["--arr1", "1,1,0,0", "--arr2", "2,3,0,0"])
        model = RecurrentCNN(args.arr1, args.arr2, convnext_version=2)
        self.assertEqual(model.stage_depths, (1, 1, 0, 0))
        self.assertEqual(model.stage_repeats, (2, 3, 0, 0))

        suffix = architecture_suffix(
            model.stage_depths, model.stage_repeats, convnext_version=2
        )
        self.assertEqual(
            suffix,
            "convnextV2_ARR1-1-1-0-0_ARR2-2-3-0-0_REG-0-0-0-0",
        )
        self.assertEqual(
            append_architecture_suffix(
                "outputs/custom",
                model.stage_depths,
                model.stage_repeats,
                convnext_version=2,
            ),
            f"outputs/custom_{suffix}",
        )
        self.assertEqual(
            append_architecture_suffix(
                "my-wandb-project",
                model.stage_depths,
                model.stage_repeats,
                convnext_version=2,
            ),
            f"my-wandb-project_{suffix}",
        )
        self.assertTrue(
            architecture_suffix(
                model.stage_depths,
                model.stage_repeats,
                convnext_version=1,
            ).startswith("convnextV1_")
        )

        reg_suffix = architecture_suffix(
            model.stage_depths,
            model.stage_repeats,
            (1, 0, 0, 0),
            (2, 3, 4, 5),
            convnext_version=2,
        )
        self.assertEqual(
            reg_suffix,
            "convnextV2_ARR1-1-1-0-0_ARR2-2-3-0-0_REG-1-0-0-0_NREG-2-3-4-5",
        )
        args = parse_args([
            "--reg-mode", "1,0,0,0",
            "--n-reg", "2,3,4,5",
            "--delta-mode",
            "--reg-head",
        ])
        self.assertEqual(args.reg_mode, "1,0,0,0")
        self.assertEqual(args.n_reg, "2,3,4,5")
        self.assertTrue(args.delta_mode)
        self.assertTrue(args.reg_head)
        variant_suffix = architecture_suffix(
            model.stage_depths,
            model.stage_repeats,
            (1, 0, 0, 0),
            (2, 3, 4, 5),
            convnext_version=2,
            delta_mode=True,
            reg_head=True,
        )
        self.assertTrue(variant_suffix.endswith("_DELTA1_REGHEAD1"))

        atto_args = parse_args(["--conv-model", "A"])
        self.assertEqual(atto_args.conv_model, "a")
        atto_suffix = architecture_suffix(
            model.stage_depths,
            model.stage_repeats,
            convnext_version=2,
            conv_model="a",
        )
        self.assertTrue(atto_suffix.startswith("convnextV2A_"))
        with self.assertRaisesRegex(ValueError, "requires ConvNeXt V2"):
            architecture_suffix(
                model.stage_depths,
                model.stage_repeats,
                convnext_version=1,
                conv_model="a",
            )
        with self.assertRaisesRegex(ValueError, "does not support RATS"):
            architecture_suffix(
                model.stage_depths,
                model.stage_repeats,
                reg_mode=(1, 0, 0, 0),
                convnext_version=2,
                conv_model="a",
            )

    def test_epoch_environment_defaults_and_generated_names(self):
        with patch.dict(os.environ, {"EPOCHS": "120", "WARMUP_EPOCHS": "7"}, clear=False):
            args = parse_args([])
        args.world_size = 1
        self.assertEqual(args.epochs, 120)
        self.assertEqual(args.warmup_epochs, 7)
        self.assertFalse(args.strict_official_recipe)
        self.assertEqual(default_run_name(args), "convnext-official-ep120-warmup7")
        output_dir = default_output_dir(args, (1, 1, 2, 0), (3, 3, 6, 0))
        self.assertIn("_ep120_warmup7_", output_dir)

        atto_args = parse_args(["--conv-model", "a"])
        atto_args.world_size = 1
        self.assertEqual(
            default_run_name(atto_args),
            "convnextV2A-official-ep300-warmup20",
        )
        self.assertIn(
            "convnextV2A_",
            default_output_dir(atto_args, (1, 1, 2, 0), (3, 3, 6, 0)),
        )

    def test_command_line_epochs_override_environment(self):
        with patch.dict(os.environ, {"EPOCHS": "120", "WARMUP_EPOCHS": "7"}, clear=False):
            args = parse_args(["--epochs", "80", "--warmup-epochs", "5"])
        self.assertEqual(args.epochs, 80)
        self.assertEqual(args.warmup_epochs, 5)

    def test_recipe_and_model_exactness_are_reported_separately(self):
        args = parse_args(["--epochs", "300", "--warmup-epochs", "20"])
        args.world_size = 1
        args.rank = 0
        args.device = torch.device("cpu")
        recurrent = RecurrentCNN(
            (1, 0, 0, 0),
            (1, 0, 0, 0),
            convnext_version=2,
            drop_path_rate=0.1,
        )
        config = build_config(args, recurrent, 1000, 1_281_167, 50_000, 312)
        self.assertTrue(config["training_recipe_exact"])
        self.assertFalse(config["paper_model_exact"])
        self.assertEqual(config["training"]["effective_batch_size"], 4096)
        self.assertEqual(config["training"]["peak_lr"], 4e-3)
        self.assertEqual(config["architecture"]["reg_mode"], [0, 0, 0, 0])
        self.assertEqual(config["architecture"]["register_applications"], 0)
        self.assertFalse(config["architecture"]["delta_mode"])
        self.assertFalse(config["architecture"]["reg_head"])
        self.assertEqual(config["architecture"]["conv_model"], "t")
        self.assertEqual(config["architecture"]["stage_widths"], list(TINY_STAGE_WIDTHS))
        self.assertEqual(config["architecture"]["last_width"], 96)

        atto = RecurrentCNN(
            (1, 1, 1, 0),
            (1, 2, 3, 0),
            convnext_version=2,
            conv_model="a",
        )
        atto_config = build_config(args, atto, 1000, 1_281_167, 50_000, 312)
        self.assertEqual(atto_config["architecture"]["conv_model"], "a")
        self.assertEqual(
            atto_config["architecture"]["stage_widths"], list(ATTO_STAGE_WIDTHS)
        )
        self.assertEqual(atto_config["architecture"]["last_width"], 160)
        self.assertFalse(atto_config["paper_model_exact"])

    def test_old_resume_arguments_default_to_tiny(self):
        args = parse_args([])
        saved = {
            key: getattr(args, key)
            for key in RESUME_ARGUMENT_KEYS
            if key != "conv_model"
        }
        args.resume = "checkpoint.pt"
        _restore_resume_arguments(args, {"arguments": saved})
        self.assertEqual(args.conv_model, "t")

    def test_optimizer_loss_and_update_scheduler(self):
        args = parse_args([
            "--batch-size", "2",
            "--grad-accum-steps", "1",
            "--reference-batch-size", "2",
            "--epochs", "3",
            "--warmup-epochs", "1",
            "--no-strict-official-recipe",
        ])
        args.world_size = 1
        model = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2))
        optimizer = create_optimizer(args, model)
        self.assertEqual(sorted({group["weight_decay"] for group in optimizer.param_groups}), [0.0, 0.05])
        scheduler = create_update_scheduler(args, optimizer, updates_per_epoch=2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], args.warmup_lr)
        scheduler.step_update(2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], args.base_lr)
        scheduler.step_update(6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], args.min_lr)
        self.assertIsInstance(create_train_criterion(args), SoftTargetCrossEntropy)

    def test_distributed_eval_sampler_has_no_duplicates(self):
        shards = [
            list(DistributedEvalSampler(range(5), rank, 2)) for rank in range(2)
        ]
        self.assertEqual(sorted(item for shard in shards for item in shard), list(range(5)))

    def test_cpu_smoke_and_resume_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "imagenet"
            for split, count in (("train", 1), ("val", 1)):
                for class_index in range(2):
                    class_dir = data_root / split / str(class_index)
                    class_dir.mkdir(parents=True)
                    for image_index in range(count):
                        value = 40 + class_index * 80 + image_index
                        Image.new("RGB", (20, 20), (value, value, value)).save(
                            class_dir / f"{image_index}.png"
                        )

            output_base = root / "output"
            architecture = (
                "convnextV2A_ARR1-1-0-0-0_ARR2-1-0-0-0_REG-0-0-0-0"
            )
            output_dir = root / f"output_{architecture}"
            main([
                "--data-root", str(data_root),
                "--output-dir", str(output_base),
                "--device", "cpu",
                "--no-amp",
                "--no-strict-official-recipe",
                "--smoke",
                "--image-size", "16",
                "--arr1", "1,0,0,0",
                "--arr2", "1,0,0,0",
                "--conv-model", "a",
                "--wandb-mode", "disabled",
            ])
            latest = output_dir / "checkpoint_latest.pt"
            self.assertTrue(latest.is_file())
            self.assertTrue((output_dir / "checkpoint_best.pt").is_file())
            self.assertTrue((output_dir / "checkpoint_best_raw.pt").is_file())
            self.assertTrue((output_dir / "checkpoint_final.pt").is_file())
            config = json.loads((output_dir / "config.json").read_text())
            self.assertFalse(config["training_recipe_exact"])
            self.assertEqual(config["architecture"]["conv_model"], "a")
            self.assertEqual(config["architecture"]["stage_widths"], [40, 80, 160, 320])
            self.assertEqual(config["architecture"]["last_width"], 40)
            self.assertEqual(
                config["training"]["wandb_project"],
                f"recurrent-convnext-imagenet_{architecture}",
            )
            records = (output_dir / "metrics.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(records), 1)
            record = json.loads(records[0])
            self.assertIn("raw", record)
            self.assertIn("ema", record)
            self.assertEqual(record["best_raw_acc1"], record["raw"]["acc1"])
            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["best_raw_acc1"], record["raw"]["acc1"])

            main([
                "--data-root", str(data_root),
                "--device", "cpu",
                "--arr1", "1,0,0,0",
                "--arr2", "1,0,0,0",
                "--conv-model", "a",
                "--image-size", "16",
                "--batch-size", "2",
                "--grad-accum-steps", "1",
                "--epochs", "1",
                "--warmup-epochs", "0",
                "--no-amp",
                "--no-strict-official-recipe",
                "--workers", "0",
                "--resume", str(latest),
                "--wandb-mode", "disabled",
            ])
            self.assertEqual(
                len((output_dir / "metrics.jsonl").read_text().strip().splitlines()),
                1,
            )

if __name__ == "__main__":
    unittest.main()
