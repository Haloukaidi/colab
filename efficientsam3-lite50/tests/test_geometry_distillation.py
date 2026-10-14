"""Regression checks for the SA-1B point/box geometry distillation phase."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch

from lite50.distillation import unpack_find_stage_geometry
from lite50.train_policy import apply_train_policy, build_optimizer, assert_optimizer_coverage


def _stage(points=None, point_mask=None, boxes=None, box_mask=None):
    batch_size = 2
    if points is None:
        points = torch.zeros(batch_size, 0, 257)
    if point_mask is None:
        point_mask = torch.zeros(points.shape[:2], dtype=torch.bool)
    if boxes is None:
        boxes = torch.zeros(0, batch_size, 4)
    if box_mask is None:
        box_mask = torch.zeros(batch_size, boxes.shape[0], dtype=torch.bool)
    return SimpleNamespace(
        input_points=points, input_points_mask=point_mask,
        input_boxes=boxes, input_boxes_mask=box_mask,
        input_boxes_label=torch.ones(boxes.shape[:2], dtype=torch.long),
    )


class PromptConversionTests(unittest.TestCase):
    def test_point_only_normalized_coordinates_and_positive_negative_labels(self):
        raw = torch.tensor([
            [[.25, .50, 1.], [.75, .35, 0.]],
            [[.20, .10, 1.], [0., 0., 0.]],
        ])
        mask = torch.tensor([[False, False], [False, True]])
        packed = unpack_find_stage_geometry(
            _stage(points=raw, point_mask=mask), require_geometry=True
        )
        self.assertEqual(tuple(packed["point_embeddings"].shape), (2, 2, 2))
        self.assertEqual(tuple(packed["point_labels"].shape), (2, 2))
        self.assertEqual(int(packed["point_labels"][1, 0]), 0)
        self.assertTrue(packed["point_mask"][1, 1])
        self.assertIsNone(packed["box_embeddings"])
        self.assertTrue(torch.equal(packed["point_embeddings"][0, 0], raw[0, 0, :2]))

    def test_box_only_is_not_mistaken_for_points(self):
        boxes = torch.tensor([[[.5, .5, .3, .3], [.25, .25, .2, .2]]])
        packed = unpack_find_stage_geometry(_stage(boxes=boxes), require_geometry=True)
        self.assertIsNone(packed["point_embeddings"])
        self.assertTrue(torch.equal(packed["box_embeddings"], boxes))

    def test_mixed_batch_padding_preserves_prompt_identity(self):
        points = torch.tensor([
            [[.2, .3, 1.]],  # item 0: point
            [[0., 0., 0.]],  # item 1: no point
        ])
        boxes = torch.tensor([[
            [0., 0., 0., 0.],  # item 0: no box
            [.5, .5, .2, .2],  # item 1: box
        ]])
        packed = unpack_find_stage_geometry(
            _stage(points=points, point_mask=torch.tensor([[False], [True]]),
                   boxes=boxes, box_mask=torch.tensor([[True], [False]])),
            require_geometry=True,
        )
        self.assertEqual(tuple(packed["point_embeddings"].shape), (1, 2, 2))
        self.assertTrue(packed["box_mask"][0, 0])
        self.assertTrue(packed["point_mask"][1, 0])
        self.assertFalse(packed["box_mask"][1, 0])

    def test_reject_incompatible_collator_or_empty_geometry(self):
        with self.assertRaisesRegex(ValueError, "input_points_embedding_dim=3"):
            unpack_find_stage_geometry(
                _stage(points=torch.zeros(2, 1, 257)),
                require_geometry=True,
            )
        with self.assertRaisesRegex(ValueError, "requires a real"):
            unpack_find_stage_geometry(_stage(), require_geometry=True)
        self.assertIsNone(unpack_find_stage_geometry(_stage())["point_embeddings"])

    def test_reject_out_of_range_valid_point(self):
        with self.assertRaisesRegex(ValueError, "Invalid normalized"):
            unpack_find_stage_geometry(
                _stage(points=torch.tensor([[[1.5, .1, 1.]], [[.2, .3, 1.]]])),
                require_geometry=True,
            )


class PhasePolicyTests(unittest.TestCase):
    def test_geometry_updates_projection_fpn_only(self):
        from test_distillation import dummy_model
        model = dummy_model()
        report = apply_train_policy(model, "geometry")
        self.assertEqual(set(report["groups"]), {"projection", "fpn0", "fpn1", "fpn2"})
        self.assertFalse(any(p.requires_grad for p in model.backbone.language_backbone.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.geometry_encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.transformer.encoder.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.segmentation_head.parameters()))
        optimizer = build_optimizer(model)
        assert_optimizer_coverage(model, optimizer)
        self.assertTrue(model.backbone.vision_backbone.trunk.model.head.training)
        self.assertFalse(model.backbone.vision_backbone.convs[3].training)


@unittest.skipUnless(os.environ.get("SAM3_ROOT"), "Requires pinned official SAM3 imports")
class RealGeometryBatchTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import pycocotools.mask as mask_utils
        from PIL import Image
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for directory in ("manifests", "sa1b/images/train"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), "blue").save(self.root / "sa1b/images/train/sa_123.jpg")
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[3:8, 4:10] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        image = {
            "image_id": "sa_123", "split": "train", "image_path": "sa1b/images/train/sa_123.jpg",
            "width": 16, "height": 16,
        }
        (self.root / "manifests/images.jsonl").write_text(json.dumps(image) + "\n")
        base = {
            **image, "segmentation": rle, "mask_area": int(mask.sum()),
            "bbox_xyxy_px": [4., 3., 10., 8.],
            "point_xy_px": [6.5, 5.5], "point_label": 1,
        }
        rows = [{**base, "region_id": "7"}, {**base, "region_id": "8"}]
        (self.root / "manifests/geometry.jsonl").write_text(
            "\n".join(json.dumps(x) for x in rows) + "\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_official_transforms_and_mixed_collator(self):
        from lite50.geometry_data import SA1BGeometryStage3Dataset, collate_geometry
        ds = SA1BGeometryStage3Dataset(self.root, split="train", prompt_mode="mixed")
        a, b = ds[0], ds[1]
        self.assertIsNotNone(a.find_queries[0].input_points)
        self.assertIsNone(a.find_queries[0].input_bbox)
        self.assertIsNone(b.find_queries[0].input_points)
        self.assertIsNotNone(b.find_queries[0].input_bbox)
        packed = collate_geometry([a, b])["all"]
        self.assertEqual(tuple(packed.img_batch.shape), (2, 3, 1008, 1008))
        self.assertEqual(tuple(packed.find_inputs[0].input_points.shape), (2, 1, 3))
        self.assertEqual(tuple(packed.find_inputs[0].input_boxes.shape), (1, 2, 4))
        self.assertTrue(packed.find_targets[0].segments.any())
        prompt = unpack_find_stage_geometry(packed.find_inputs[0], require_geometry=True)
        self.assertEqual(int(prompt["point_labels"][0, 0]), 1)
        self.assertFalse(prompt["point_mask"][0, 0])
        self.assertTrue(prompt["point_mask"][1, 0])
        self.assertTrue(prompt["box_mask"][0, 0])
        self.assertFalse(prompt["box_mask"][1, 0])
        self.assertAlmostEqual(float(prompt["point_embeddings"][0, 0, 0]), 6.5 / 16, places=5)
        self.assertAlmostEqual(float(prompt["point_embeddings"][0, 0, 1]), 5.5 / 16, places=5)

    def test_one_mode_per_epoch_rotation(self):
        from lite50.geometry_data import SA1BGeometryStage3Dataset
        ds = SA1BGeometryStage3Dataset(self.root, split="train", prompt_mode="mixed")
        self.assertIsNotNone(ds[0].find_queries[0].input_points)
        ds.set_epoch(1)
        self.assertIsNotNone(ds[0].find_queries[0].input_bbox)


if __name__ == "__main__":
    unittest.main()
