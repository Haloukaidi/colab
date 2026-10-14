"""Reject wrong SACap joins instead of training on plausible-looking masks."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
from pycocotools import mask as mask_utils


@unittest.skipUnless(os.environ.get("SAM3_ROOT"), "Set SAM3_ROOT to the official sam3 directory")
class NativeSACapTests(unittest.TestCase):
    def setUp(self):
        from lite50.train_data import SACapStage3Dataset
        self.dataset_class = SACapStage3Dataset
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for folder in ("manifests", "sacap", "sa1b/images/train", "sa1b/annotations/train"):
            (self.root / folder).mkdir(parents=True)
        Image.new("RGB", (32, 32), "red").save(self.root / "sa1b/images/train/sa_1.jpg")
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[2:6, 2:6] = 1
        rle = mask_utils.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
        (self.root / "sa1b/annotations/train/sa_1.json").write_text(json.dumps(
            {"annotations": [{"id": 7, "segmentation": rle}]}))
        image = {"image_id": "sa_1", "split": "train", "image_path": "sa1b/images/train/sa_1.jpg"}
        (self.root / "manifests/images.jsonl").write_text(json.dumps(image)+"\n")
        pq.write_table(pa.Table.from_pylist([{"imagename": "sa_1.jpg", "caption": "red image",
            "segments_info": [{"anno_id": 7, "caption": "a red square"}]}]), self.root / "sacap/train.parquet")
        self.row = {**image, "region_id": "7", "segmentation": rle,
            "bbox_xyxy_px": [2, 2, 6, 6], "source_text": "a red square", "short_text": "red square",
            "enabled_for_training": True, "review_status": "accepted", "eligible_automatic_candidate": False}
        self.manifest = self.root / "prompts.jsonl"
        self.write([self.row])

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, rows):
        self.manifest.write_text("\n".join(json.dumps(r) for r in rows)+"\n")

    def load(self, **kwargs):
        return self.dataset_class(self.root, self.manifest, resolution=32, **kwargs)

    def test_native_schema_and_synchronized_targets(self):
        dp = self.load(prompt_mode="text_box")[0]
        self.assertEqual(dp.find_queries[0].query_text, "red square")
        self.assertFalse(dp.find_queries[0].is_exhaustive)
        self.assertEqual(int(dp.images[0].objects[0].segment.sum()), 16)
        np.testing.assert_allclose(dp.find_queries[0].input_bbox.numpy(), [[.125, .125, .125, .125]])

    def test_caption_or_region_mismatch_rejected(self):
        self.write([{**self.row, "source_text": "a different object"}])
        with self.assertRaisesRegex(ValueError, "annotation / prompt"):
            self.load()
        self.write([{**self.row, "region_id": "8"}])
        with self.assertRaises(ValueError):
            self.load()

    def test_review_rejection_cannot_be_overridden(self):
        self.write([{**self.row, "review_status": "hold", "eligible_automatic_candidate": True}])
        with self.assertRaisesRegex(ValueError, "No approved"):
            self.load(include_unreviewed=True)

    def test_duplicate_sidecar_rejected(self):
        self.write([self.row, self.row])
        with self.assertRaisesRegex(ValueError, "Duplicate prompt"):
            self.load()
