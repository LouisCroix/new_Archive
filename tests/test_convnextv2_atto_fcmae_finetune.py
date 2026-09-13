import copy
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import timm
import torch
from PIL import Image
from safetensors.torch import save_file

from imagenet_recurrent_convnextv2_atto_fcmae_finetune import (
    ATTO_DEPTHS,
    MODEL_ID,
    OFFICIAL_RECIPE,
    RecurrentConvNeXtV2Atto,
    create_model,
    effective_batch_size,
    expanded_drop_path_rates,
    layer_id_for_parameter,
    main,
    parameter_groups,
    parse_args,
    parse_stage_repeats,
    recipe_mismatches,
    resolve_grad_accum_steps,
    scaled_peak_lr,
)


class ConvNeXtV2AttoFCMAETest(unittest.TestCase):
    def test_stage_repeat_parser(self):
        self.assertEqual(parse_stage_repeats("1,2,3,4"), (1, 2, 3, 4))
        for invalid in ("1,2,3", "1,0,1,1", "1,2,x,4"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_stage_repeats(invalid)

    def test_expanded_drop_path_schedule(self):
        native = expanded_drop_path_rates((1, 1, 1, 1), 0.1)
        self.assertEqual(len(native), sum(ATTO_DEPTHS))
        self.assertEqual(native[0], 0.0)
        self.assertAlmostEqual(native[-1], 0.1)
        recurrent = expanded_drop_path_rates((2, 1, 3, 1), 0.2)
        self.assertEqual(len(recurrent), 2 * 2 + 2 + 6 * 3 + 2)
        self.assertTrue(all(a < b for a, b in zip(recurrent, recurrent[1:])))

        base = timm.create_model(MODEL_ID, pretrained=False, num_classes=3)
        model = RecurrentConvNeXtV2Atto(base, (2, 1, 1, 1), 0.2)
        observed = []
        with patch(
            "imagenet_recurrent_convnextv2_atto_fcmae_finetune.drop_path",
            side_effect=lambda tensor, probability, training: observed.append(probability) or tensor,
        ):
            model(torch.randn(1, 3, 32, 32))
        self.assertEqual(observed, list(model.drop_path_rates))

    def test_native_repeat_one_is_logit_equivalent(self):
        torch.manual_seed(4)
        native = timm.create_model(MODEL_ID, pretrained=False, num_classes=7, drop_path_rate=0.1)
        recurrent = RecurrentConvNeXtV2Atto(copy.deepcopy(native), (1, 1, 1, 1), 0.1)
        native.eval()
        recurrent.eval()
        inputs = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            torch.testing.assert_close(recurrent(inputs), native(inputs), rtol=0, atol=0)

    def test_recurrence_keeps_parameters_and_downsamples_once(self):
        base = timm.create_model(MODEL_ID, pretrained=False, num_classes=5)
        native_count = sum(parameter.numel() for parameter in base.parameters())
        recurrent = RecurrentConvNeXtV2Atto(base, (2, 3, 1, 2), 0.1)
        self.assertEqual(sum(parameter.numel() for parameter in recurrent.parameters()), native_count)
        calls = [0, 0, 0, 0]
        hooks = [stage.downsample.register_forward_hook(lambda _m, _i, _o, index=index: calls.__setitem__(index, calls[index] + 1)) for index, stage in enumerate(recurrent.stages)]
        recurrent.eval()
        with torch.no_grad():
            recurrent(torch.randn(1, 3, 32, 32))
        for hook in hooks:
            hook.remove()
        self.assertEqual(calls, [1, 1, 1, 1])

    def test_local_fcmae_state_and_head_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            source = timm.create_model(MODEL_ID, pretrained=False, num_classes=0)
            path = Path(directory) / "model.safetensors"
            save_file(source.state_dict(), str(path))
            model, metadata = create_model(pretrained=False, pretrained_checkpoint=str(path), num_classes=11)
            self.assertEqual(model.head.fc.weight.shape, (11, 320))
            self.assertLess(abs(model.head.fc.weight.std().item() - 2e-5), 3e-6)
            self.assertEqual(model.head.fc.bias.abs().max().item(), 0.0)
            self.assertEqual(metadata["source"], str(path.resolve()))
            for key, value in source.state_dict().items():
                torch.testing.assert_close(model.state_dict()[key], value)

    def test_official_single_layer_decay_mapping_and_groups(self):
        self.assertEqual(layer_id_for_parameter("stem.0.weight"), 1)
        self.assertEqual(layer_id_for_parameter("stages.0.blocks.0.conv_dw.weight"), 1)
        self.assertEqual(layer_id_for_parameter("stages.0.blocks.1.conv_dw.weight"), 2)
        self.assertEqual(layer_id_for_parameter("stages.1.downsample.1.weight"), 3)
        self.assertEqual(layer_id_for_parameter("stages.2.blocks.5.mlp.fc2.weight"), 10)
        self.assertEqual(layer_id_for_parameter("stages.3.blocks.1.conv_dw.weight"), 12)
        self.assertEqual(layer_id_for_parameter("head.fc.weight"), 13)
        base = timm.create_model(MODEL_ID, pretrained=False, num_classes=3)
        model = RecurrentConvNeXtV2Atto(base)
        groups = parameter_groups(model, 0.3, 0.9)
        self.assertTrue(any(group["weight_decay"] == 0.0 for group in groups))
        self.assertTrue(any(group["weight_decay"] == 0.3 for group in groups))
        head_group = next(group for group in groups if group["group_name"] == "layer_13_decay")
        self.assertEqual(head_group["lr_scale"], 1.0)

    def test_recipe_defaults_and_overrides(self):
        args = parse_args([])
        args.world_size = 1
        self.assertEqual(resolve_grad_accum_steps(args), 32)
        self.assertEqual(effective_batch_size(args), OFFICIAL_RECIPE["global_batch_size"])
        self.assertAlmostEqual(scaled_peak_lr(args), 8e-4)
        self.assertEqual(recipe_mismatches(args), [])
        args.epochs = 2
        self.assertEqual(len(recipe_mismatches(args)), 1)
        self.assertIn("epochs", recipe_mismatches(args)[0])

    def test_automatic_accumulation_respects_global_batch_cap(self):
        args = parse_args([
            "--batch-size", "64", "--max-global-batch-size", "1000",
        ])
        args.world_size = 4
        self.assertEqual(resolve_grad_accum_steps(args), 3)
        self.assertEqual(effective_batch_size(args), 768)
        with self.assertRaisesRegex(ValueError, "smaller than"):
            too_small = parse_args([
                "--batch-size", "256", "--max-global-batch-size", "1000",
            ])
            too_small.world_size = 4
            resolve_grad_accum_steps(too_small)

    def test_cpu_smoke_and_resume_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "imagenet"
            for split in ("train", "val"):
                for class_index in range(2):
                    class_dir = data_root / split / str(class_index)
                    class_dir.mkdir(parents=True)
                    for image_index in range(2 if split == "train" else 1):
                        Image.new("RGB", (40, 40), (40 + class_index * 100, 30, 20)).save(class_dir / f"{image_index}.png")
            output = root / "output"
            main([
                "--data-root", str(data_root), "--output-dir", str(output),
                "--device", "cpu", "--no-amp", "--smoke", "--image-size", "32",
                "--wandb-mode", "disabled",
            ])
            latest = output / "checkpoint_latest.pt"
            self.assertTrue(latest.is_file())
            self.assertTrue((output / "checkpoint_best.pt").is_file())
            self.assertTrue((output / "checkpoint_best_raw.pt").is_file())
            self.assertTrue((output / "checkpoint_final.pt").is_file())
            config = json.loads((output / "config.json").read_text())
            self.assertEqual(config["architecture"]["stage_repeats"], [1, 1, 1, 1])
            self.assertFalse(config["pretrained"]["pretrained"])
            main(["--resume", str(latest), "--device", "cpu", "--wandb-mode", "disabled"])
            self.assertEqual(len((output / "metrics.jsonl").read_text().strip().splitlines()), 1)

    def test_launcher_dry_run(self):
        script = Path(__file__).parents[1] / "scripts" / "run_imagenet_recurrent_convnextv2_atto_fcmae_finetune.sh"
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env.update({
                "PROJECT_ROOT": str(Path(__file__).parents[1]),
                "DATA_ROOT": directory,
                "PYTHON_BIN": os.sys.executable,
                "DRY_RUN": "1",
                "REQUIRE_CUDA": "0",
                "WANDB_MODE": "disabled",
                "STAGE_REPEATS": "1,1,1,1",
            })
            result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=True)
        self.assertIn("stage_repeats=1,1,1,1", result.stdout)
        self.assertIn("effective_batch=1024", result.stdout)
        self.assertIn("torch.distributed.run", result.stdout)


if __name__ == "__main__":
    unittest.main()
