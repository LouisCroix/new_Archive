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
    DistributedEvalSampler,
    append_architecture_suffix,
    architecture_suffix,
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
from recurrent_cnn import RecurrentCNN


class RecurrentCNNOfficialTest(unittest.TestCase):
    def test_default_arguments_match_official_recipe(self):
        with patch.dict(os.environ, {"EPOCHS": "300", "WARMUP_EPOCHS": "20"}):
            args = parse_args([])
        args.world_size = 1
        self.assertEqual(effective_batch_size(args), 4096)
        self.assertEqual(scaled_peak_lr(args), 4e-3)
        self.assertEqual(official_recipe_mismatches(args), [])
        self.assertEqual(args.convnext_version, 2)
        self.assertEqual(args.arr1, "1,1,2,0")
        self.assertEqual(args.arr2, "3,3,6,0")
        self.assertEqual(args.reg_mode, "0,0,0,0")
        self.assertEqual(args.n_reg, "8,8,8,8")
        self.assertFalse(args.delta_mode)
        self.assertFalse(args.reg_head)

    def test_architecture_arguments_and_destination_suffixes(self):
        args = parse_args(["--arr1", "2,3,0,0", "--arr2", "4,5,0,0"])
        model = RecurrentCNN(args.arr1, args.arr2, convnext_version=2)
        self.assertEqual(model.stage_depths, (2, 3, 0, 0))
        self.assertEqual(model.stage_repeats, (4, 5, 0, 0))

        suffix = architecture_suffix(model.stage_depths, model.stage_repeats)
        self.assertEqual(suffix, "ARR1-2-3-0-0_ARR2-4-5-0-0")
        self.assertEqual(
            append_architecture_suffix("outputs/custom", model.stage_depths, model.stage_repeats),
            f"outputs/custom_{suffix}",
        )
        self.assertEqual(
            append_architecture_suffix("my-wandb-project", model.stage_depths, model.stage_repeats),
            f"my-wandb-project_{suffix}",
        )

        reg_suffix = architecture_suffix(
            model.stage_depths,
            model.stage_repeats,
            (1, 0, 0, 0),
            (2, 3, 4, 5),
        )
        self.assertEqual(
            reg_suffix,
            "ARR1-2-3-0-0_ARR2-4-5-0-0_REG-1-0-0-0_NREG-2-3-4-5",
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
            delta_mode=True,
            reg_head=True,
        )
        self.assertTrue(variant_suffix.endswith("_DELTA1_REGHEAD1"))

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
            (1, 1, 2, 0),
            (3, 3, 6, 0),
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

        args.convnext_version = 1
        native = RecurrentCNN(
            (3, 3, 9, 3),
            (1, 1, 1, 1),
            convnext_version=1,
            drop_path_rate=0.1,
        )
        native_config = build_config(args, native, 1000, 1_281_167, 50_000, 312)
        self.assertTrue(native_config["paper_model_exact"])

        registered = RecurrentCNN(
            (3, 3, 9, 3),
            (1, 1, 1, 1),
            convnext_version=1,
            reg_mode=(0, 0, 1, 0),
        )
        registered_config = build_config(
            args, registered, 1000, 1_281_167, 50_000, 312
        )
        self.assertFalse(registered_config["paper_model_exact"])
        self.assertEqual(registered_config["architecture"]["register_applications"], 1)

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
        dataset = list(range(11))
        shards = [list(DistributedEvalSampler(dataset, rank, 3)) for rank in range(3)]
        flattened = [item for shard in shards for item in shard]
        self.assertEqual(sorted(flattened), dataset)
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_cpu_smoke_and_resume_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "imagenet"
            for split, count in (("train", 2), ("val", 1)):
                for class_index in range(2):
                    class_dir = data_root / split / str(class_index)
                    class_dir.mkdir(parents=True)
                    for image_index in range(count):
                        value = 40 + class_index * 80 + image_index
                        Image.new("RGB", (48, 48), (value, value, value)).save(
                            class_dir / f"{image_index}.png"
                        )

            output_base = root / "output"
            architecture = "ARR1-1-0-0-0_ARR2-1-0-0-0"
            output_dir = root / f"output_{architecture}"
            main([
                "--data-root", str(data_root),
                "--output-dir", str(output_base),
                "--device", "cpu",
                "--no-amp",
                "--no-strict-official-recipe",
                "--smoke",
                "--image-size", "32",
                "--arr1", "1,0,0,0",
                "--arr2", "1,0,0,0",
                "--wandb-mode", "disabled",
            ])
            latest = output_dir / "checkpoint_latest.pt"
            self.assertTrue(latest.is_file())
            self.assertTrue((output_dir / "checkpoint_best.pt").is_file())
            self.assertTrue((output_dir / "checkpoint_final.pt").is_file())
            config = json.loads((output_dir / "config.json").read_text())
            self.assertFalse(config["training_recipe_exact"])
            self.assertEqual(
                config["training"]["wandb_project"],
                f"recurrent-convnext-imagenet_{architecture}",
            )
            records = (output_dir / "metrics.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(records), 1)
            record = json.loads(records[0])
            self.assertIn("raw", record)
            self.assertIn("ema", record)

            main([
                "--data-root", str(data_root),
                "--device", "cpu",
                "--arr1", "1,0,0,0",
                "--arr2", "1,0,0,0",
                "--image-size", "32",
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

            legacy_path = output_dir / "checkpoint_legacy_v6.pt"
            legacy = torch.load(latest, map_location="cpu", weights_only=False)
            legacy["arguments"].pop("reg_mode")
            legacy["arguments"].pop("n_reg")
            for key in (
                "reg_mode",
                "n_reg",
                "register_stage_count",
                "register_applications",
                "register_attention",
                "register_heads",
                "register_sdpa_backend",
                "register_mlp_ratio",
                "register_data_term",
                "register_reconstruction",
                "register_layerscale",
            ):
                legacy["config"]["architecture"].pop(key)
            torch.save(legacy, legacy_path)
            main([
                "--data-root", str(data_root),
                "--device", "cpu",
                "--arr1", "1,0,0,0",
                "--arr2", "1,0,0,0",
                "--image-size", "32",
                "--batch-size", "2",
                "--grad-accum-steps", "1",
                "--epochs", "1",
                "--warmup-epochs", "0",
                "--no-amp",
                "--no-strict-official-recipe",
                "--workers", "0",
                "--resume", str(legacy_path),
                "--wandb-mode", "disabled",
            ])

            register_architecture = (
                "ARR1-1-0-0-0_ARR2-1-0-0-0_"
                "REG-1-0-0-0_NREG-2-8-8-8"
            )
            register_output = root / f"register_output_{register_architecture}"
            main([
                "--data-root", str(data_root),
                "--output-dir", str(root / "register_output"),
                "--device", "cpu",
                "--no-amp",
                "--no-strict-official-recipe",
                "--smoke",
                "--image-size", "32",
                "--arr1", "1,0,0,0",
                "--arr2", "1,0,0,0",
                "--reg-mode", "1,0,0,0",
                "--n-reg", "2,8,8,8",
                "--wandb-mode", "disabled",
            ])
            register_config = json.loads(
                (register_output / "config.json").read_text()
            )
            self.assertEqual(
                register_config["architecture"]["reg_mode"], [1, 0, 0, 0]
            )
            self.assertEqual(
                register_config["architecture"]["register_applications"], 1
            )
            self.assertTrue(
                register_config["model_arch"].endswith("array_tied_rats_v7")
            )

            with self.assertRaisesRegex(ValueError, "Resume arguments differ"):
                main([
                    "--data-root", str(data_root),
                    "--device", "cpu",
                    "--arr1", "1,0,0,0",
                    "--arr2", "1,0,0,0",
                    "--image-size", "32",
                    "--batch-size", "1",
                    "--grad-accum-steps", "1",
                    "--epochs", "1",
                    "--warmup-epochs", "0",
                    "--no-amp",
                    "--no-strict-official-recipe",
                    "--workers", "0",
                    "--resume", str(latest),
                    "--wandb-mode", "disabled",
                ])


if __name__ == "__main__":
    unittest.main()
