import unittest

import torch
import torch.nn as nn

from recurrent_cnn import (
    IterationBatchNorm2d,
    RecurrentCNN,
    adamw_parameter_groups,
)


class RecurrentCNNTest(unittest.TestCase):
    def test_resnet_keeps_one_running_statistics_bank_per_iteration(self):
        model = RecurrentCNN("resnet", iterations=3, num_classes=10).train()
        output, residuals = model(torch.randn(4, 3, 32, 32), log_residuals=True)
        self.assertEqual(output.shape, (4, 10))
        self.assertEqual(len(residuals), 3)

        norms = [
            module for module in model.block.modules()
            if isinstance(module, IterationBatchNorm2d)
        ]
        self.assertEqual(len(norms), 2)
        for norm in norms:
            self.assertEqual(tuple(norm.running_mean.shape), (3, 64))
            self.assertEqual(tuple(norm.running_var.shape), (3, 64))
            self.assertEqual(norm.num_batches_tracked.tolist(), [1, 1, 1])
            self.assertFalse(torch.equal(norm.running_mean[0], norm.running_mean[2]))

        model.eval()
        with torch.no_grad():
            eval_output, _ = model(torch.randn(4, 3, 32, 32))
        self.assertEqual(eval_output.shape, (4, 10))

    def test_resnet_eval_matches_training_statistics_after_calibration(self):
        torch.manual_seed(1)
        model = RecurrentCNN("resnet", iterations=3, num_classes=10)
        inputs = torch.randn(16, 3, 32, 32)
        model.train()
        with torch.no_grad():
            for _ in range(100):
                train_output, _ = model(inputs)
            train_output, _ = model(inputs)
        model.eval()
        with torch.no_grad():
            eval_output, _ = model(inputs)
        self.assertLess((train_output - eval_output).abs().mean().item(), 0.01)

    def test_convnext_layernorm_has_identical_train_and_eval_behavior(self):
        model = RecurrentCNN("convnext", iterations=3, num_classes=10)
        inputs = torch.randn(2, 3, 32, 32)
        model.train()
        with torch.no_grad():
            train_output, _ = model(inputs)
        model.eval()
        with torch.no_grad():
            eval_output, _ = model(inputs)
        torch.testing.assert_close(train_output, eval_output)
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in model.modules()))

    def test_adamw_groups_filter_1d_parameters_and_bias_only(self):
        for block_type in ("resnet", "convnext"):
            model = RecurrentCNN(block_type, iterations=2, num_classes=10)
            groups = adamw_parameter_groups(model, weight_decay=0.05)
            grouped = {
                id(parameter): group["weight_decay"]
                for group in groups for parameter in group["params"]
            }
            self.assertEqual(len(grouped), len(list(model.parameters())))
            for name, parameter in model.named_parameters():
                expected = 0.0 if parameter.ndim == 1 or name.endswith(".bias") else 0.05
                self.assertEqual(grouped[id(parameter)], expected, name)
            if block_type == "convnext":
                self.assertEqual(grouped[id(model.block.layer_scale)], 0.05)

    def test_normalization_fix_does_not_add_trainable_parameters(self):
        self.assertEqual(
            sum(parameter.numel() for parameter in RecurrentCNN("resnet", 12).parameters()),
            148_520,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in RecurrentCNN("convnext", 12).parameters()),
            181_384,
        )

    def test_mode_and_block_depth_parameter_counts(self):
        expected = {
            ("naive", "resnet", 1): 148_520,
            ("naive", "resnet", 2): 222_504,
            ("naive", "convnext", 1): 181_384,
            ("naive", "convnext", 2): 260_680,
            ("pro", "resnet", 1): 3_039_784,
            ("pro", "resnet", 2): 4_220_456,
            ("pro", "convnext", 1): 3_118_408,
            ("pro", "convnext", 2): 4_320_328,
        }
        for (mode, block_type, depth), count in expected.items():
            model = RecurrentCNN(
                block_type,
                iterations=2,
                num_classes=1000,
                mode=mode,
                block_depth=depth,
            )
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), count)

    def test_pro_modes_use_third_stage_width_and_depth(self):
        for block_type, width in (("resnet", 256), ("convnext", 384)):
            model = RecurrentCNN(
                block_type,
                iterations=2,
                num_classes=10,
                mode="pro",
                block_depth=2,
            ).train()
            frontend_shapes = []
            handle = model.frontend.register_forward_hook(
                lambda _module, _inputs, output: frontend_shapes.append(tuple(output.shape))
            )
            output, residuals = model(torch.randn(2, 3, 64, 64), log_residuals=True)
            handle.remove()
            self.assertEqual(frontend_shapes, [(2, width, 4, 4)])
            self.assertEqual(output.shape, (2, 10))
            self.assertEqual(len(residuals), 2)
            self.assertIsInstance(model.block, nn.Sequential)
            self.assertEqual(len(model.block), 2)

    def test_resnet_deep_cell_has_iteration_stats_for_every_block(self):
        model = RecurrentCNN(
            "resnet", iterations=3, num_classes=10, mode="pro", block_depth=2
        ).train()
        model(torch.randn(2, 3, 64, 64))
        norms = [
            module for module in model.block.modules()
            if isinstance(module, IterationBatchNorm2d)
        ]
        self.assertEqual(len(norms), 4)
        for norm in norms:
            self.assertEqual(norm.num_batches_tracked.tolist(), [1, 1, 1])

    def test_invalid_mode_and_depth_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported mode"):
            RecurrentCNN(mode="unknown")
        with self.assertRaisesRegex(ValueError, "BLOCK_DEPTH"):
            RecurrentCNN(block_depth=0)


if __name__ == "__main__":
    unittest.main()
