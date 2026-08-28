import unittest
from dataclasses import asdict
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from timm.utils import ModelEmaV3

from peq_timm_common import (
    CHECKPOINT_FORMAT_VERSION,
    MODEL_FAMILY,
    ModelConfig,
    Net,
    model_name_suffix,
    parse_n_reg_schedule,
    run_name_suffix,
    training_name_suffix,
    train_one_epoch,
    validate_checkpoint_compatibility,
    wandb_project_name,
)


def tiny_config(**overrides):
    values = {
        "image_size": 8,
        "patch_size": 4,
        "dim": 12,
        "n_reg_schedule": (2, 3),
        "heads": 3,
        "steps": 2,
        "num_classes": 5,
        "delta_backend": "naive",
        "sdpa_backend": "auto",
    }
    values.update(overrides)
    return ModelConfig(**values)


class DeltaPeqModelTests(unittest.TestCase):
    def test_parser_and_schedule_normalization(self):
        self.assertEqual(parse_n_reg_schedule("[2, 3]"), (2, 3))
        self.assertEqual(tiny_config(n_reg_schedule=(4,)).n_reg_schedule, (4, 4))
        with self.assertRaises(ValueError):
            parse_n_reg_schedule("2,,3")
        with self.assertRaises(ValueError):
            tiny_config(n_reg_schedule=(1, 2, 3))

    def test_all_modes_and_attention_layouts_forward_and_backward(self):
        image = torch.randn(1, 3, 8, 8)
        for attention in ("sequential", "rats"):
            for mode in ("single", "tied", "untied", "tied_data", "tied_data_rec"):
                with self.subTest(attention=attention, mode=mode):
                    model = Net(tiny_config(attention=attention, mode=mode))
                    output, _, recon = model(image)
                    self.assertEqual(output.shape, (1, 5))
                    self.assertEqual(len(recon), 1 if mode == "single" else 2)
                    output.sum().backward()

    def test_all_readouts(self):
        image = torch.randn(2, 3, 8, 8)
        for readout in ("reg", "weighted", "patch", "sum", "concat"):
            with self.subTest(readout=readout):
                model = Net(tiny_config(attention="rats", readout=readout))
                output, resid, recon = model(image, log=True)
                self.assertEqual(output.shape, (2, 5))
                self.assertEqual(len(resid), 2)
                self.assertEqual(len(recon), 2)

    def test_dynamic_registers_and_layerscale_layout(self):
        tied = Net(tiny_config(layerscale=True, steps=2))
        self.assertEqual(tied.r0.shape, (1, 3, 12))
        self.assertEqual(tied.block.ls_p.shape, (2, 12))
        untied = Net(tiny_config(layerscale=True, mode="untied", steps=2))
        self.assertTrue(all(block.ls_p.shape == (12,) for block in untied.blocks))

    def test_invalid_model_options(self):
        invalid = (
            {"attention": "softmax"},
            {"delta_backend": "missing"},
            {"delta_chunk_size": 8},
            {"readout": "missing"},
        )
        for override in invalid:
            with self.subTest(override=override), self.assertRaises(ValueError):
                tiny_config(**override)

    def test_training_loop_protocol_is_unchanged(self):
        model = Net(tiny_config(mode="tied"))
        ema = ModelEmaV3(model, decay=0.9, foreach=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        loader = DataLoader(
            TensorDataset(torch.randn(2, 3, 8, 8), torch.tensor([0, 1])),
            batch_size=2,
        )
        args = SimpleNamespace(
            device=torch.device("cpu"),
            grad_accum_steps=1,
            amp=False,
            amp_dtype="bfloat16",
            clip_grad=None,
            mode="tied",
            lrec=0.3,
        )
        metrics, updates = train_one_epoch(
            args,
            model,
            loader,
            optimizer,
            nn.CrossEntropyLoss(),
            None,
            ema,
            None,
            0,
        )
        self.assertEqual(updates, 1)
        self.assertIn("loss", metrics)

    def test_experiment_names_cover_model_and_training_identity(self):
        args = SimpleNamespace(
            seed=0,
            amp=True,
            amp_dtype="bfloat16",
            lrec=0.3,
            opt="lamb",
            lr=1e-3,
            min_lr=1e-5,
            weight_decay=0.03,
            epochs=400,
            warmup_epochs=5,
            batch_size=256,
            world_size=2,
            grad_accum_steps=2,
        )
        cfg = tiny_config(
            image_size=224,
            patch_size=16,
            dim=384,
            heads=6,
            steps=12,
            n_reg_schedule=(64,),
            delta_backend="fla",
            sdpa_backend="flash",
        )
        suffix = run_name_suffix(args, cfg)
        model_suffix = model_name_suffix(cfg)
        training_suffix = training_name_suffix(args)
        project = wandb_project_name("peq_imagenet_pretrain", cfg)
        self.assertLessEqual(len(project), 128)
        self.assertIn("sequential_deltafla_sdpaflash", project)
        self.assertIn(model_suffix, project)
        self.assertNotIn("lr0.001", project)
        self.assertIn("lr0.001", training_suffix)
        changed_args = SimpleNamespace(**{**vars(args), "weight_decay": 0.05})
        changed_suffix = run_name_suffix(changed_args, cfg)
        changed_project = wandb_project_name("peq_imagenet_pretrain", cfg)
        self.assertNotEqual(suffix, changed_suffix)
        self.assertEqual(project, changed_project)
        changed_cfg = tiny_config(**{**asdict(cfg), "readout": "weighted"})
        self.assertNotEqual(
            project, wandb_project_name("peq_imagenet_pretrain", changed_cfg)
        )


class DeltaPeqCheckpointTests(unittest.TestCase):
    def test_new_checkpoint_metadata_roundtrip(self):
        cfg = tiny_config(attention="rats")
        checkpoint = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "model_family": MODEL_FAMILY,
            "model_arch": cfg.model_arch,
            "model_config": asdict(cfg),
        }
        self.assertEqual(
            validate_checkpoint_compatibility(checkpoint, "memory"), cfg
        )

    def test_legacy_and_standalone_checkpoints_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "legacy or foreign"):
            validate_checkpoint_compatibility(
                {"format_version": 1, "model_config": {}}, "legacy.pt"
            )
        with self.assertRaisesRegex(ValueError, "standalone delta_peq.py"):
            validate_checkpoint_compatibility(
                {"config": {"model_family": MODEL_FAMILY}}, "delta.pt"
            )


if __name__ == "__main__":
    unittest.main()
