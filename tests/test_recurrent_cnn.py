from functools import partial
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from torchvision.models.convnext import CNBlock, LayerNorm2d

from deltanet import DeltaNet
from recurrent_cnn import (
    ConvNeXtV2Block,
    GlobalResponseNorm,
    RATSRegisterBlock,
    RecurrentCNN,
    RepeatedStage,
    ScheduledCNBlock,
    adamw_parameter_groups,
    legacy_config_to_stage_arrays,
    load_resume_config,
    migrate_legacy_state_dict,
    validate_convnext_version,
    validate_register_arrays,
    validate_stage_arrays,
)


def make_block(width):
    return CNBlock(
        width,
        layer_scale=1e-6,
        stochastic_depth_prob=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


class LegacyPromini(nn.Module):
    """Minimal v5 layout used to verify exact checkpoint migration."""

    def __init__(self, repeats=2, num_classes=10):
        super().__init__()
        norm = partial(LayerNorm2d, eps=1e-6)
        self.stem = nn.Sequential(nn.Conv2d(3, 96, 4, 4), norm(96))
        self.frontend = nn.Sequential(
            LegacyRepeatedStage(nn.Sequential(make_block(96)), repeats),
            nn.Sequential(norm(96), nn.Conv2d(96, 192, 2, 2)),
            LegacyRepeatedStage(nn.Sequential(make_block(192)), repeats),
            nn.Sequential(norm(192), nn.Conv2d(192, 384, 2, 2)),
        )
        self.block = make_block(384)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_norm = nn.LayerNorm(384, eps=1e-6)
        self.head = nn.Linear(384, num_classes)

    def forward(self, inputs):
        outputs = self.frontend(self.stem(inputs))
        for _ in range(self.frontend[0].repeats):
            outputs = self.block(outputs)
        outputs = self.pool(outputs).flatten(1)
        return self.head(self.head_norm(outputs))


class LegacyRepeatedStage(nn.Module):
    def __init__(self, stage, repeats):
        super().__init__()
        self.stage = stage
        self.repeats = repeats

    def forward(self, inputs):
        outputs = inputs
        for _ in range(self.repeats):
            outputs = self.stage(outputs)
        return outputs


class RecurrentCNNTest(unittest.TestCase):
    def test_register_array_validation(self):
        self.assertEqual(
            validate_register_arrays("0,1,1,0", "2,3,4,5", (1, 1, 1, 0)),
            ((0, 1, 1, 0), (2, 3, 4, 5)),
        )
        invalid = (
            ((0, 1, 0), (8, 8, 8, 8), (1, 1, 0, 0)),
            ((0, 2, 0, 0), (8, 8, 8, 8), (1, 1, 0, 0)),
            ((0, 1, 0, 0), (8, 0, 8, 8), (1, 1, 0, 0)),
            ((0, 0, 1, 0), (8, 8, 8, 8), (1, 1, 0, 0)),
        )
        for reg_mode, n_reg, depths in invalid:
            with self.subTest(reg_mode=reg_mode, n_reg=n_reg, depths=depths):
                with self.assertRaises(ValueError):
                    validate_register_arrays(reg_mode, n_reg, depths)

    def test_zero_register_mode_is_exactly_backward_compatible(self):
        torch.manual_seed(11)
        implicit = RecurrentCNN((1, 1, 1, 0), (2, 2, 2, 0), num_classes=10).eval()
        torch.manual_seed(11)
        explicit = RecurrentCNN(
            (1, 1, 1, 0),
            (2, 2, 2, 0),
            num_classes=10,
            reg_mode=(0, 0, 0, 0),
            n_reg=(2, 3, 4, 5),
        ).eval()
        self.assertEqual(tuple(implicit.state_dict()), tuple(explicit.state_dict()))
        self.assertFalse(any("register_block" in key for key in implicit.state_dict()))
        inputs = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            expected, _ = implicit(inputs)
            actual, _ = explicit(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_v6_checkpoint_defaults_to_disabled_registers(self):
        model = RecurrentCNN((1, 0, 0, 0), (2, 0, 0, 0), num_classes=10)
        checkpoint = {
            "config": {
                "model_family": "recurrent_cnn",
                "experiment_version": 6,
                "convnext_version": 1,
                "arr1": [1, 0, 0, 0],
                "arr2": [2, 0, 0, 0],
            },
            "model": model.state_dict(),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pt")
            torch.save(checkpoint, path)
            with patch.dict(os.environ, {}, clear=False):
                loaded = load_resume_config(path)
        self.assertEqual(loaded["reg_mode"], (0, 0, 0, 0))
        self.assertEqual(loaded["n_reg"], (8, 8, 8, 8))
        self.assertFalse(loaded["delta_mode"])
        self.assertFalse(loaded["reg_head"])
        self.assertEqual(tuple(loaded["model"]), tuple(model.state_dict()))

    def test_delta_mode_applies_to_every_enabled_register_stage(self):
        kwargs = dict(
            stage_depths=(1, 1, 0, 0),
            stage_repeats=(2, 3, 0, 0),
            num_classes=7,
            convnext_version=2,
            reg_mode=(1, 1, 0, 0),
            n_reg=(2, 3, 8, 8),
        )
        rats = RecurrentCNN(**kwargs)
        model = RecurrentCNN(**kwargs, delta_mode=True)
        stages = [module for module in model.features if isinstance(module, RepeatedStage)]
        self.assertTrue(all(stage.register_block.delta_mode for stage in stages))
        self.assertTrue(
            all(isinstance(stage.register_block.patch_delta, DeltaNet) for stage in stages)
        )

        expected_delta_parameters = sum(
            sum(parameter.numel() for parameter in stage.register_block.patch_delta.parameters())
            + sum(parameter.numel() for parameter in stage.register_block.lnp.parameters())
            for stage in stages
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters())
            - sum(parameter.numel() for parameter in rats.parameters()),
            expected_delta_parameters,
        )

        calls = [0, 0]
        handles = []
        for index, stage in enumerate(stages):
            def count(_module, _inputs, _output, stage_index=index):
                calls[stage_index] += 1
            handles.append(stage.register_block.patch_delta.register_forward_hook(count))
        output, _ = model(torch.randn(2, 3, 32, 32))
        output.sum().backward()
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, [2, 3])
        self.assertIsNotNone(stages[0].register_block.patch_delta.q.weight.grad)
        self.assertIsNotNone(stages[1].register_block.patch_delta.o.weight.grad)

    def test_reg_head_concatenates_final_register_and_feature_means(self):
        model = RecurrentCNN(
            (1, 0, 0, 0),
            (2, 0, 0, 0),
            num_classes=7,
            reg_mode=(1, 0, 0, 0),
            n_reg=(3, 8, 8, 8),
            reg_head=True,
        ).eval()
        stage = model.features[0]
        captured = {}

        def capture_registers(_module, _inputs, output):
            captured["registers"] = output[1]

        def capture_pool(_module, _inputs, output):
            captured["features"] = output.flatten(1)

        def capture_readout(_module, inputs):
            captured["readout"] = inputs[0]

        handles = [
            stage.register_block.register_forward_hook(capture_registers),
            model.pool.register_forward_hook(capture_pool),
            model.head_norm.register_forward_pre_hook(capture_readout),
        ]
        output, _ = model(torch.randn(2, 3, 32, 32))
        for handle in handles:
            handle.remove()
        expected = torch.cat(
            (captured["registers"].mean(dim=1), captured["features"]), dim=-1
        )
        torch.testing.assert_close(captured["readout"], expected)
        self.assertEqual(model.head.in_features, 192)
        self.assertEqual(output.shape, (2, 7))

    def test_delta_and_register_head_require_compatible_register_stages(self):
        with self.assertRaisesRegex(ValueError, "delta_mode"):
            RecurrentCNN(delta_mode=True)
        with self.assertRaisesRegex(ValueError, "final active stage"):
            RecurrentCNN(
                (1, 1, 0, 0),
                (1, 1, 0, 0),
                reg_mode=(1, 0, 0, 0),
                reg_head=True,
            )

    def test_current_delta_register_head_parameter_count(self):
        model = RecurrentCNN(
            (1, 1, 1, 0),
            (3, 3, 6, 0),
            num_classes=1000,
            convnext_version=2,
            reg_mode=(0, 0, 1, 0),
            n_reg=(8, 8, 64, 8),
            delta_mode=True,
            reg_head=True,
        )
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 5_431_016)

    def test_stage_registers_are_independent_and_repeated(self):
        model = RecurrentCNN(
            (1, 1, 0, 0),
            (2, 3, 0, 0),
            num_classes=7,
            convnext_version=2,
            reg_mode=(1, 1, 0, 0),
            n_reg=(2, 5, 8, 8),
        )
        stages = [module for module in model.features if isinstance(module, RepeatedStage)]
        self.assertIsInstance(stages[0].register_block, RATSRegisterBlock)
        self.assertIsInstance(stages[1].register_block, RATSRegisterBlock)
        self.assertIsNot(stages[0].register_block, stages[1].register_block)
        self.assertEqual(stages[0].register_block.r0.shape, (1, 2, 96))
        self.assertEqual(stages[1].register_block.r0.shape, (1, 5, 192))
        self.assertEqual(model.register_stage_count, 2)
        self.assertEqual(model.register_applications, 5)

        calls = [0, 0]
        seen_registers = [[], []]
        handles = []
        for index, stage in enumerate(stages):
            def record(_module, inputs, _output, stage_index=index):
                calls[stage_index] += 1
                seen_registers[stage_index].append(inputs[1].detach().clone())
            handles.append(stage.register_block.register_forward_hook(record))
        output, residuals = model(torch.randn(2, 3, 32, 32), log_residuals=True)
        output.sum().backward()
        for handle in handles:
            handle.remove()
        self.assertEqual(output.shape, (2, 7))
        self.assertEqual(calls, [2, 3])
        self.assertFalse(torch.equal(seen_registers[0][0], seen_registers[0][1]))
        self.assertFalse(torch.equal(seen_registers[1][0], seen_registers[1][1]))
        self.assertEqual({key: len(value) for key, value in residuals.items()}, {
            "stage1": 2,
            "stage2": 3,
        })
        self.assertIsNotNone(stages[0].register_block.r0.grad)
        self.assertIsNotNone(stages[1].register_block.w1.weight.grad)

    def test_convnext_versions_select_the_expected_block(self):
        v1 = RecurrentCNN((1, 0, 0, 0), (1, 0, 0, 0), num_classes=10)
        v2 = RecurrentCNN(
            (1, 0, 0, 0),
            (1, 0, 0, 0),
            num_classes=10,
            convnext_version=2,
        )
        v1_block = v1.features[0].stage[0]
        v2_block = v2.features[0].stage[0]
        self.assertIsInstance(v1_block, CNBlock)
        self.assertIsInstance(v2_block, ConvNeXtV2Block)
        self.assertIsInstance(v2_block.grn, GlobalResponseNorm)
        self.assertFalse(hasattr(v2_block, "layer_scale"))
        self.assertTrue(torch.count_nonzero(v2_block.grn.gamma) == 0)
        self.assertTrue(torch.count_nonzero(v2_block.grn.beta) == 0)
        with torch.no_grad():
            output, _ = v2.eval()(torch.randn(2, 3, 32, 32))
        self.assertEqual(output.shape, (2, 10))

    def test_convnext_v2_native_parameter_count(self):
        model = RecurrentCNN(
            (3, 3, 9, 3),
            (1, 1, 1, 1),
            convnext_version=2,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            28_635_496,
        )

    def test_convnext_version_validation(self):
        self.assertEqual(validate_convnext_version("1"), 1)
        self.assertEqual(validate_convnext_version(2), 2)
        for invalid in (0, 3, "v2", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_convnext_version(invalid)

    def test_native_topology_matches_convnext_tiny_parameter_count(self):
        model = RecurrentCNN((3, 3, 9, 3), (1, 1, 1, 1))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 28_589_128)
        self.assertEqual(model.active_stages, 4)
        self.assertEqual(model.last_width, 768)
        self.assertEqual(model.unique_blocks, 18)
        self.assertEqual(model.block_applications, 18)
        self.assertEqual(len(model.features), 7)
        self.assertEqual(model.head.in_features, 768)

    def test_native_tiny_drop_path_matches_unrolled_linear_schedule(self):
        model = RecurrentCNN(
            (3, 3, 9, 3),
            (1, 1, 1, 1),
            drop_path_rate=0.1,
        )
        self.assertEqual(len(model.drop_path_rates), 18)
        self.assertEqual(model.drop_path_rates[0], 0.0)
        self.assertAlmostEqual(model.drop_path_rates[-1], 0.1)
        self.assertTrue(all(
            left < right
            for left, right in zip(model.drop_path_rates, model.drop_path_rates[1:])
        ))
        flattened = tuple(
            probability
            for stage in model.features
            if isinstance(stage, RepeatedStage)
            for probability in stage.drop_path_probs
        )
        self.assertEqual(flattened, model.drop_path_rates)

    def test_recurrent_drop_path_varies_across_shared_block_calls(self):
        model = RecurrentCNN(
            (1, 1, 2, 0),
            (3, 3, 6, 0),
            convnext_version=2,
            drop_path_rate=0.1,
        )
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 3_557_032)
        self.assertEqual(len(model.drop_path_rates), 18)
        stages = [module for module in model.features if isinstance(module, RepeatedStage)]
        self.assertEqual([len(stage.drop_path_probs) for stage in stages], [3, 3, 12])
        self.assertEqual(len(set(stages[0].drop_path_probs)), 3)
        self.assertEqual(len(set(stages[2].drop_path_probs)), 12)

    def test_drop_path_preserves_state_dict_and_eval_outputs(self):
        plain = RecurrentCNN((1, 1, 1, 0), (2, 2, 2, 0), num_classes=10).eval()
        regularized = RecurrentCNN(
            (1, 1, 1, 0),
            (2, 2, 2, 0),
            num_classes=10,
            drop_path_rate=0.1,
        ).eval()
        regularized.load_state_dict(plain.state_dict(), strict=True)
        self.assertEqual(tuple(plain.state_dict()), tuple(regularized.state_dict()))
        self.assertIsInstance(regularized.features[0].stage[0], ScheduledCNBlock)
        inputs = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            expected, _ = plain(inputs)
            actual, _ = regularized(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_promini_topology_and_repetition_counts(self):
        model = RecurrentCNN((1, 1, 1, 0), (3, 3, 3, 0), num_classes=10).eval()
        self.assertEqual(sum(parameter.numel() for parameter in RecurrentCNN().parameters()), 2_347_720)
        calls = [0, 0, 0]
        handles = []
        stages = [module for module in model.features if isinstance(module, RepeatedStage)]
        for index, stage in enumerate(stages):
            def count(_module, _inputs, _output, stage_index=index):
                calls[stage_index] += 1
            handles.append(stage.stage.register_forward_hook(count))
        with torch.no_grad():
            output, residuals = model(torch.randn(1, 3, 64, 64), log_residuals=True)
        for handle in handles:
            handle.remove()
        self.assertEqual(output.shape, (1, 10))
        self.assertEqual(calls, [3, 3, 3])
        self.assertEqual({key: len(value) for key, value in residuals.items()}, {
            "stage1": 3,
            "stage2": 3,
            "stage3": 3,
        })

    def test_active_prefix_controls_downsampling_and_head_width(self):
        for active_stages, expected_width in enumerate((96, 192, 384, 768), start=1):
            depths = tuple(1 if index < active_stages else 0 for index in range(4))
            repeats = depths
            model = RecurrentCNN(depths, repeats, num_classes=7).eval()
            self.assertEqual(model.active_stages, active_stages)
            self.assertEqual(model.last_width, expected_width)
            self.assertEqual(model.head.in_features, expected_width)
            self.assertEqual(len(model.features), 2 * active_stages - 1)
            with torch.no_grad():
                output, _ = model(torch.randn(1, 3, 64, 64))
            self.assertEqual(output.shape, (1, 7))

    def test_repeats_do_not_change_parameter_count(self):
        once = RecurrentCNN((2, 2, 2, 0), (1, 1, 1, 0))
        repeated = RecurrentCNN((2, 2, 2, 0), (12, 7, 3, 0))
        self.assertEqual(
            sum(parameter.numel() for parameter in once.parameters()),
            sum(parameter.numel() for parameter in repeated.parameters()),
        )

    def test_array_validation(self):
        valid = validate_stage_arrays("3,3,9,3", "1,1,1,1")
        self.assertEqual(valid, ((3, 3, 9, 3), (1, 1, 1, 1)))
        invalid = (
            ((1, 1, 1), (1, 1, 1)),
            ((1, -1, 0, 0), (1, 1, 0, 0)),
            ((1, 1, 0, 0), (1, 0, 0, 0)),
            ((1, 0, 1, 0), (1, 0, 1, 0)),
            ((0, 0, 0, 0), (0, 0, 0, 0)),
        )
        for depths, repeats in invalid:
            with self.subTest(depths=depths, repeats=repeats):
                with self.assertRaises(ValueError):
                    validate_stage_arrays(depths, repeats)

    def test_legacy_mode_mapping(self):
        expected = {
            "naive": ((2, 0, 0, 0), (5, 0, 0, 0)),
            "pro": ((3, 3, 2, 0), (1, 1, 5, 0)),
            "promax": ((3, 3, 2, 0), (5, 5, 5, 0)),
            "promini": ((1, 1, 2, 0), (5, 5, 5, 0)),
        }
        for mode, arrays in expected.items():
            config = {"mode": mode, "block_depth": 2, "T": 5}
            self.assertEqual(legacy_config_to_stage_arrays(config), arrays)

    def test_v5_model_optimizer_and_scheduler_migrate_exactly(self):
        torch.manual_seed(4)
        legacy = LegacyPromini(repeats=2, num_classes=10).eval()
        legacy_optimizer = torch.optim.AdamW(
            adamw_parameter_groups(legacy, 0.05), lr=5e-4, weight_decay=0.0
        )
        legacy_scheduler = torch.optim.lr_scheduler.LambdaLR(
            legacy_optimizer, lambda step: 1.0 - 0.01 * step
        )
        inputs = torch.randn(2, 3, 32, 32)
        legacy.train()
        loss = legacy(inputs).square().mean()
        loss.backward()
        legacy_optimizer.step()
        legacy_scheduler.step()
        legacy.eval()

        config = {
            "experiment_version": 5,
            "mode": "promini",
            "block_depth": 1,
            "T": 2,
        }
        migrated = RecurrentCNN((1, 1, 1, 0), (2, 2, 2, 0), num_classes=10).eval()
        migrated.load_state_dict(
            migrate_legacy_state_dict(legacy.state_dict(), config), strict=True
        )
        migrated_optimizer = torch.optim.AdamW(
            adamw_parameter_groups(migrated, 0.05), lr=5e-4, weight_decay=0.0
        )
        migrated_optimizer.load_state_dict(legacy_optimizer.state_dict())
        migrated_scheduler = torch.optim.lr_scheduler.LambdaLR(
            migrated_optimizer, lambda step: 1.0 - 0.01 * step
        )
        migrated_scheduler.load_state_dict(legacy_scheduler.state_dict())

        with torch.no_grad():
            expected = legacy(inputs)
            actual, _ = migrated(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(migrated_scheduler.state_dict(), legacy_scheduler.state_dict())
        self.assertEqual(len(migrated_optimizer.state), len(legacy_optimizer.state))

    def test_layernorm_and_adamw_parameter_groups(self):
        model = RecurrentCNN((1, 0, 0, 0), (2, 0, 0, 0), num_classes=10)
        inputs = torch.randn(2, 3, 32, 32)
        model.train()
        with torch.no_grad():
            train_output, _ = model(inputs)
        model.eval()
        with torch.no_grad():
            eval_output, _ = model(inputs)
        torch.testing.assert_close(train_output, eval_output)
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in model.modules()))

        groups = adamw_parameter_groups(model, weight_decay=0.05)
        grouped = {
            id(parameter): group["weight_decay"]
            for group in groups for parameter in group["params"]
        }
        self.assertEqual(len(grouped), len(list(model.parameters())))
        for name, parameter in model.named_parameters():
            expected_decay = 0.0 if parameter.ndim == 1 or name.endswith(".bias") else 0.05
            self.assertEqual(grouped[id(parameter)], expected_decay, name)


if __name__ == "__main__":
    unittest.main()
