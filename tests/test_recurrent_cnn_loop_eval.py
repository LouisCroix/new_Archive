import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image

from imagenet_recurrent_cnn_loop_eval import (
    DistributedEvalSampler,
    RESULT_CSV,
    RESULT_JSON,
    RESULT_PNG,
    build_sweep_model,
    checkpoint_spec,
    current_stage_repeats,
    load_official_checkpoint,
    main,
    set_stage3_repeats,
    sweep_repeat_counts,
)
from recurrent_cnn import CONV_MODEL_STAGE_WIDTHS, RecurrentCNN


def make_checkpoint(
    root: Path,
    *,
    depths=(1, 1, 1, 0),
    repeats=(1, 1, 1, 0),
    num_classes=2,
    image_size=16,
    conv_model="t",
):
    model = RecurrentCNN(
        depths,
        repeats,
        num_classes=num_classes,
        convnext_version=2,
        drop_path_rate=0.1,
        conv_model=conv_model,
    )
    checkpoint = {
        "format_version": 1,
        "model_family": "recurrent_cnn_official",
        "epoch": 4,
        "num_updates": 50,
        "model": model.state_dict(),
        "config": {
            "architecture": {
                "arr1": list(depths),
                "arr2": list(repeats),
                "convnext_version": 2,
                "conv_model": conv_model,
                "stage_widths": list(CONV_MODEL_STAGE_WIDTHS[conv_model]),
                "last_width": CONV_MODEL_STAGE_WIDTHS[conv_model][
                    sum(depth > 0 for depth in depths) - 1
                ],
                "drop_path_rate": 0.1,
                "reg_mode": [0, 0, 0, 0],
                "n_reg": [8, 8, 8, 8],
                "delta_mode": False,
                "reg_head": False,
            },
            "training": {
                "image_size": image_size,
                "data_root": str(root / "imagenet"),
            },
            "dataset": {"num_classes": num_classes},
        },
    }
    path = root / "checkpoint.pt"
    torch.save(checkpoint, path)
    return path, checkpoint, model


def make_validation_dataset(root: Path, num_classes=2):
    for class_index in range(num_classes):
        class_dir = root / "imagenet" / "val" / str(class_index)
        class_dir.mkdir(parents=True)
        value = 40 + class_index * 100
        Image.new("RGB", (24, 24), (value, value, value)).save(
            class_dir / "sample.png"
        )


class RecurrentCNNLoopEvalTest(unittest.TestCase):
    def test_sweep_range_and_checkpoint_architecture(self):
        self.assertEqual(sweep_repeat_counts(10), list(range(1, 21)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, checkpoint, _ = make_checkpoint(
                root, repeats=(2, 3, 10, 0)
            )
            loaded = load_official_checkpoint(path)
            spec = checkpoint_spec(loaded)
            self.assertEqual(spec.conv_model, "t")
            self.assertEqual(spec.stage_depths, (1, 1, 1, 0))
            self.assertEqual(spec.training_stage_repeats, (2, 3, 10, 0))
            self.assertEqual(spec.maximum_stage_repeats, (2, 3, 20, 0))
            self.assertEqual(checkpoint["model"].keys(), loaded["model"].keys())
            inactive = dict(checkpoint)
            inactive["config"] = {
                **checkpoint["config"],
                "architecture": {
                    **checkpoint["config"]["architecture"],
                    "arr1": [1, 1, 0, 0],
                    "arr2": [1, 1, 0, 0],
                },
            }
            with self.assertRaisesRegex(ValueError, "Stage 3 must be enabled"):
                checkpoint_spec(inactive)

            malformed_weights = dict(checkpoint["model"])
            malformed_weights.pop("head.weight")
            with self.assertRaisesRegex(ValueError, "Raw checkpoint weights"):
                build_sweep_model(spec, malformed_weights)

            legacy = dict(checkpoint)
            legacy_architecture = dict(checkpoint["config"]["architecture"])
            legacy_architecture.pop("conv_model")
            legacy_architecture.pop("stage_widths")
            legacy_architecture.pop("last_width")
            legacy["config"] = {
                **checkpoint["config"],
                "architecture": legacy_architecture,
            }
            self.assertEqual(checkpoint_spec(legacy).conv_model, "t")

            malformed_widths = dict(checkpoint)
            malformed_widths["config"] = {
                **checkpoint["config"],
                "architecture": {
                    **checkpoint["config"]["architecture"],
                    "stage_widths": [40, 80, 160, 320],
                },
            }
            with self.assertRaisesRegex(ValueError, "stage_widths"):
                checkpoint_spec(malformed_widths)

            wrong_family = dict(checkpoint)
            wrong_family["model_family"] = "recurrent_cnn"
            wrong_path = root / "wrong.pt"
            torch.save(wrong_family, wrong_path)
            with self.assertRaisesRegex(ValueError, "model_family"):
                load_official_checkpoint(wrong_path)

    def test_training_repeat_logits_match_and_other_stages_remain_fixed(self):
        torch.manual_seed(0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, checkpoint, training_model = make_checkpoint(
                root, repeats=(2, 3, 2, 0), conv_model="a"
            )
            spec = checkpoint_spec(checkpoint)
            self.assertEqual(spec.stage_widths, (40, 80, 160, 320))
            sweep_model = build_sweep_model(spec, checkpoint["model"])
            training_model.eval()
            sweep_model.eval()
            set_stage3_repeats(sweep_model, spec.training_stage3_repeats)
            self.assertEqual(current_stage_repeats(sweep_model), (2, 3, 2))
            inputs = torch.randn(1, 3, 16, 16)
            with torch.inference_mode():
                expected, _ = training_model(inputs)
                actual, _ = sweep_model(inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

            set_stage3_repeats(sweep_model, 4)
            self.assertEqual(current_stage_repeats(sweep_model), (2, 3, 4))
            with self.assertRaisesRegex(ValueError, "must be in"):
                set_stage3_repeats(sweep_model, 5)

    def test_distributed_sampler_has_no_duplicates(self):
        dataset = list(range(11))
        shards = [
            list(DistributedEvalSampler(dataset, rank, 3)) for rank in range(3)
        ]
        flattened = [item for shard in shards for item in shard]
        self.assertEqual(sorted(flattened), dataset)
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_cpu_smoke_outputs_and_completed_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_validation_dataset(root)
            checkpoint_path, _, _ = make_checkpoint(root, conv_model="a")
            output_dir = root / "sweep"
            arguments = [
                "--checkpoint",
                str(checkpoint_path),
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
                "--no-amp",
                "--batch-size",
                "2",
                "--workers",
                "0",
            ]
            main(arguments)
            for filename in (RESULT_JSON, RESULT_CSV, RESULT_PNG):
                self.assertTrue((output_dir / filename).is_file())
            payload = json.loads((output_dir / RESULT_JSON).read_text())
            self.assertEqual(
                [record["stage3_repeats"] for record in payload["results"]],
                [1, 2],
            )
            self.assertTrue(
                all(record["samples"] == 2 for record in payload["results"])
            )
            self.assertTrue(
                all(
                    0.0 <= record["val_acc1"] <= 100.0
                    for record in payload["results"]
                )
            )

            with patch(
                "imagenet_recurrent_cnn_loop_eval.evaluate_model",
                side_effect=AssertionError("completed points must be skipped"),
            ):
                main(arguments)
            payload["format_version"] = 2
            (output_dir / RESULT_JSON).write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "metadata does not match"):
                main(arguments)


if __name__ == "__main__":
    unittest.main()
