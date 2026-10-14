import unittest
import torch
from torch import nn
from types import SimpleNamespace
from lite50.distillation import image_mse, text_mse, text_distillation, matching_by_target, matched_geometry_kd
from lite50.train_policy import apply_train_policy, build_optimizer, assert_optimizer_coverage
from lite50.teacher_cache import cache_key, model_signature


def dummy_model():
    m = nn.Module()
    m.backbone = nn.Module()
    m.backbone.vision_backbone = nn.Module()
    v = m.backbone.vision_backbone
    v.trunk = nn.Module()
    v.trunk.model = nn.Module()
    v.trunk.model.backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    v.trunk.model.head = nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4))
    v.convs = nn.ModuleList([nn.Conv2d(4, 4, 1) for _ in range(4)])
    v.active_levels = 3
    m.backbone.language_backbone = nn.Linear(4, 4)
    m.transformer = nn.Module()
    m.transformer.encoder = nn.Module()
    m.transformer.encoder.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])
    m.transformer.decoder = nn.Linear(4, 4)
    m.geometry_encoder = nn.Linear(4, 4)
    m.segmentation_head = nn.Linear(4, 4)
    return m


class PolicyTests(unittest.TestCase):
    def test_frozen_modes_and_active_scales(self):
        m = dummy_model()
        apply_train_policy(m, "vision")
        m.eval().train()
        self.assertFalse(m.backbone.vision_backbone.trunk.model.backbone.training)
        self.assertTrue(m.backbone.vision_backbone.trunk.model.head.training)
        self.assertFalse(m.backbone.vision_backbone.convs[3].training)
        self.assertFalse(m.transformer.decoder.training)
        self.assertFalse(m.geometry_encoder.training)
        self.assertFalse(m.segmentation_head.training)
        self.assertTrue(m.training)

    def test_progressive_fusion_and_phase_reset(self):
        m = dummy_model()
        for phase, count in (("fusion1", 1), ("fusion2", 2), ("fusion4", 4), ("text", 0)):
            apply_train_policy(m, phase)
            states = [any(p.requires_grad for p in layer.parameters()) for layer in m.transformer.encoder.layers]
            self.assertEqual(states, [False]*(4-count)+[True]*count)
            self.assertFalse(any(p.requires_grad for p in m.transformer.decoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in m.backbone.vision_backbone.parameters()))

    def test_optimizer_exact_coverage(self):
        m = dummy_model()
        apply_train_policy(m, "vision")
        opt = build_optimizer(m)
        assert_optimizer_coverage(m, opt)
        opt.param_groups[0]["params"].append(next(m.geometry_encoder.parameters()))
        with self.assertRaises(RuntimeError):
            assert_optimizer_coverage(m, opt)

    def test_old_optimizer_rejected_after_unfreeze(self):
        m = dummy_model()
        apply_train_policy(m, "fusion1")
        opt = build_optimizer(m)
        apply_train_policy(m, "fusion2")
        with self.assertRaises(RuntimeError):
            assert_optimizer_coverage(m, opt)


class LossTests(unittest.TestCase):
    def test_upstream_channel_reduction_difference(self):
        self.assertEqual(float(image_mse(torch.ones(1, 3, 2, 2), torch.zeros(1, 3, 2, 2), torch.ones(1, 1, 2, 2))), 3)
        self.assertEqual(float(text_mse(torch.ones(1, 2, 3), torch.zeros(1, 2, 3), torch.ones(1, 2))), 1)

    def test_padding_has_zero_gradient(self):
        x = torch.ones(1, 4, 3, requires_grad=True)
        loss = text_mse(x, torch.zeros_like(x), torch.tensor([[1., 1., 0., 0.]]))
        loss.backward()
        self.assertEqual(int(torch.count_nonzero(x.grad[:, 2:])), 0)

    def test_different_token_masks_rejected(self):
        s = {"language_features": torch.ones(16, 1, 3), "language_mask": torch.zeros(1, 16, dtype=torch.bool)}
        t = {**s, "language_mask": torch.ones(1, 16, dtype=torch.bool)}
        with self.assertRaises(ValueError):
            text_distillation(s, t)

    def test_matching_implicit_packed_targets(self):
        self.assertEqual(matching_by_target((torch.tensor([0, 1]), torch.tensor([9, 3]), None)),
                         {(0, 0): 9, (1, 1): 3})

    def test_mask_matching_not_query_index(self):
        masks = torch.tensor([[[[10., -10.], [-10., -10.]], [[-10., -10.], [-10., 10.]]]])
        s = {"pred_masks": masks.clone().requires_grad_(),
             "indices": (torch.tensor([0, 0]), torch.tensor([0, 1]), None)}
        t = {"pred_masks": masks.flip(1),
             "indices": (torch.tensor([0, 0]), torch.tensor([1, 0]), None)}
        loss, count = matched_geometry_kd(s, t, {"masks": (masks[0] > 0).float()})
        self.assertEqual(count, 2)
        self.assertLess(float(loss.detach()), 0.001)
        loss.backward()
        self.assertIsNotNone(s["pred_masks"].grad)

    def test_wrong_teacher_mask_is_rejected(self):
        sm = torch.ones(1, 1, 2, 2, requires_grad=True)
        idx = (torch.tensor([0]), torch.tensor([0]), None)
        loss, count = matched_geometry_kd({"pred_masks": sm, "indices": idx},
            {"pred_masks": -sm.detach(), "indices": idx}, {"masks": torch.ones(1, 2, 2)})
        self.assertEqual(count, 0)
        loss.backward()
        self.assertEqual(float(sm.grad.sum()), 0)


class CacheTests(unittest.TestCase):
    def test_input_and_weight_invalidation(self):
        model = nn.BatchNorm1d(2)
        signature = model_signature(model)  # includes scalar num_batches_tracked
        inp = SimpleNamespace(img_ids=torch.tensor([0]), text_ids=torch.tensor([0]),
            input_boxes=torch.zeros(1, 1, 4), input_boxes_mask=torch.zeros(1, 1, dtype=torch.bool),
            input_boxes_label=torch.ones(1, 1, dtype=torch.long))
        batch = SimpleNamespace(img_batch=torch.zeros(1, 3, 2, 2),
                                find_text_batch=["a cat"], find_inputs=[inp])
        original = cache_key(signature, batch)
        batch.find_text_batch = ["a dog"]
        self.assertNotEqual(original, cache_key(signature, batch))
        batch.find_text_batch = ["a cat"]
        batch.img_batch[0, 0, 0, 0] = 1
        self.assertNotEqual(original, cache_key(signature, batch))
        batch.img_batch.zero_()
        inp.input_boxes[0, 0, 0] = 1
        self.assertNotEqual(original, cache_key(signature, batch))
        inp.input_boxes.zero_()
        model.running_mean.add_(1)
        self.assertNotEqual(original, cache_key(model_signature(model), batch))


if __name__ == "__main__":
    unittest.main()
