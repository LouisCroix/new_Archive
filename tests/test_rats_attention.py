import unittest

import torch

from rats_attention import RATSAttention


def manual_attention(module, stage, query, key, value):
    if stage == "compress":
        query = module.q_proj(query)
        key = module.k_proj(key)
        value = module.v_proj(value)
    elif stage == "broadcast":
        query = module.q_proj(query)
    query = module._split_heads(query)
    key = module._split_heads(key)
    value = module._split_heads(value)
    scores = query @ key.transpose(-2, -1) / module.head_dim**0.5
    weights = scores.softmax(dim=-1)
    context = weights @ value
    context = context.transpose(1, 2).contiguous().reshape(
        query.size(0), query.size(2), module.dim
    )
    return module.out_proj[stage](context), weights.mean(dim=1)


class RATSAttentionTest(unittest.TestCase):
    def test_all_stages_match_delta_peq_projection_rules(self):
        torch.manual_seed(3)
        module = RATSAttention(12, heads=3, sdpa_backend="auto")
        registers = torch.randn(2, 4, 12)
        features = torch.randn(2, 7, 12)
        operands = {
            "compress": (registers, features, features),
            "refine": (registers, registers, registers),
            "broadcast": (features, registers, registers),
        }
        for stage, values in operands.items():
            with self.subTest(stage=stage):
                expected, expected_weights = manual_attention(module, stage, *values)
                actual, actual_weights = module(
                    stage, *values, return_weights=True
                )
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual_weights, expected_weights)

    def test_sdpa_path_shapes_and_backward(self):
        module = RATSAttention(12, heads=3, sdpa_backend="auto")
        query = torch.randn(2, 5, 12, requires_grad=True)
        registers = torch.randn(2, 3, 12, requires_grad=True)
        output = module("broadcast", query, registers, registers)
        self.assertEqual(output.shape, query.shape)
        output.square().mean().backward()
        self.assertIsNotNone(query.grad)
        self.assertIsNotNone(registers.grad)
        self.assertIsNotNone(module.q_proj.weight.grad)

    def test_invalid_configuration_and_operands(self):
        with self.assertRaises(ValueError):
            RATSAttention(10, heads=3)
        with self.assertRaises(ValueError):
            RATSAttention(12, heads=3, sdpa_backend="missing")
        module = RATSAttention(12, heads=3)
        value = torch.randn(1, 2, 12)
        with self.assertRaises(ValueError):
            module("missing", value, value, value)
        with self.assertRaises(ValueError):
            module("refine", value[..., :8], value, value)


if __name__ == "__main__":
    unittest.main()
