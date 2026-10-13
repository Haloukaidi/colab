"""PyTorch-only regression tests; the official SAM3 package is not required."""
import copy
import unittest

import torch
from torch import nn

from lite50.builder import Lite50Config
from lite50.fpn_optimization import ActiveScaleFPN, enable_skip_unused_fpn


class _Level(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1)
        self.index = index
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.conv(x) + self.index


class _Trunk(nn.Module):
    def forward(self, x):
        return [x]


class _PositionEncoding(nn.Module):
    def forward(self, x):
        return torch.ones_like(x) * (1.0 / x.shape[-1])


class _OriginalNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = _Trunk()
        self.position_encoding = _PositionEncoding()
        self.convs = nn.ModuleList([_Level(i) for i in range(4)])
        self.scale_factors = (4.0, 2.0, 1.0, 0.5)
        self.sam2_convs = None

    def forward(self, x):
        x = self.trunk(x)[-1]
        features = [level(x) for level in self.convs]
        return (
            features,
            [self.position_encoding(feat) for feat in features],
            None,
            None,
        )


class _VLBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = _OriginalNeck()
        self.scalp = 1

    def forward_image(self, x):
        features, positions, _, _ = self.vision_backbone(x)
        if self.scalp:
            features = features[:-self.scalp]
            positions = positions[:-self.scalp]
        return {"backbone_fpn": features, "vision_pos_enc": positions}


class TestUnusedFPNOptimization(unittest.TestCase):
    def test_defaults(self):
        self.assertTrue(Lite50Config().skip_unused_fpn_level)
        self.assertFalse(Lite50Config(skip_unused_fpn_level=False).skip_unused_fpn_level)

    def test_exact_output_old_weights_and_gradient(self):
        torch.manual_seed(0)
        backbone = _VLBackbone()
        previous_weights = copy.deepcopy(backbone.state_dict())
        old_param_count = sum(p.numel() for p in backbone.parameters())
        image = torch.randn(1, 4, 8, 8)
        original = backbone.forward_image(image)

        self.assertEqual(len(original["backbone_fpn"]), 3)
        self.assertEqual(backbone.vision_backbone.convs[3].calls, 1)

        enable_skip_unused_fpn(backbone)
        self.assertIsInstance(backbone.vision_backbone, ActiveScaleFPN)
        self.assertEqual(backbone.scalp, 0)
        optimized = backbone.forward_image(image)

        # The unused fourth level is never invoked again.
        self.assertEqual(backbone.vision_backbone.convs[3].calls, 1)
        self.assertEqual(len(optimized["backbone_fpn"]), 3)
        for key in ("backbone_fpn", "vision_pos_enc"):
            for a, b in zip(original[key], optimized[key]):
                self.assertTrue(torch.equal(a, b))

        # Strictly load an old checkpoint without renaming/removing parameters.
        self.assertEqual(previous_weights.keys(), backbone.state_dict().keys())
        self.assertEqual(old_param_count, sum(p.numel() for p in backbone.parameters()))
        backbone.load_state_dict(previous_weights, strict=True)

        backbone.zero_grad(set_to_none=True)
        sum(y.sum() for y in backbone.forward_image(image)["backbone_fpn"]).backward()
        self.assertTrue(all(p.grad is not None for p in backbone.vision_backbone.convs[0].parameters()))
        self.assertTrue(all(p.grad is None for p in backbone.vision_backbone.convs[3].parameters()))

    def test_guards(self):
        backbone = _VLBackbone()
        backbone.scalp = 0
        with self.assertRaisesRegex(ValueError, "Expected RV-M"):
            enable_skip_unused_fpn(backbone)

        backbone = _VLBackbone()
        backbone.vision_backbone.sam2_convs = nn.ModuleList([_Level(0)])
        with self.assertRaisesRegex(ValueError, "SAM2"):
            enable_skip_unused_fpn(backbone)


if __name__ == "__main__":
    unittest.main()
