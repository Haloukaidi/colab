"""SA-1B single-point / single-box grounding samples for Lite50 geometry KD.

Reuses the official SAM3 Datapoint, DecodeRle, resize/normalize and collator.
The original mini-v1 train/val split and GT RLE masks remain authoritative.
"""
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset


def _jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


class SA1BGeometryStage3Dataset(Dataset):
    """One SA-1B region per query; point-only or box-only, never both by default.

    `mixed` rotates modes deterministically across dataset index and epoch.
    Single positive points follow the official geometry fine-tuning default.
    Box jitter / negative or multiple points can be studied separately later.
    """

    MODES = ("point", "box", "mixed")

    def __init__(self, dataset_root, split="train", resolution=1008,
                 prompt_mode="mixed", limit=None):
        from sam3.train.transforms.basic_for_api import RandomResizeAPI, ToTensorAPI, NormalizeAPI
        from sam3.train.transforms.segmentation import DecodeRle

        if split not in ("train", "val"):
            raise ValueError("Use the existing mini-v1 train/val split")
        if prompt_mode not in self.MODES:
            raise ValueError(f"prompt_mode must be one of {self.MODES}")
        if resolution != 1008:
            raise ValueError("Lite50 uses the fixed 1008 x 1008 image resolution")
        self.root = Path(dataset_root)
        self.split = split
        self.resolution = resolution
        self.prompt_mode = prompt_mode
        self.epoch = 0

        images = {
            str(row["image_id"]): row
            for row in _jsonl(self.root / "manifests/images.jsonl")
        }
        self.rows = [
            row for row in _jsonl(self.root / "manifests/geometry.jsonl")
            if row["split"] == split
        ]
        if limit is not None:
            if int(limit) < 1:
                raise ValueError("limit must be a positive integer")
            self.rows = self.rows[:int(limit)]
        if not self.rows:
            raise ValueError("No SA-1B geometry regions in requested split")

        seen = set()
        for row in self.rows:
            key = (str(row["image_id"]), str(row["region_id"]))
            if key in seen:
                raise ValueError(f"Duplicate image/region pair {key}")
            seen.add(key)
            image = images.get(key[0])
            if image is None or image["split"] != split or image["image_path"] != row["image_path"]:
                raise ValueError(f"Image not found in same split: {key}")
            h, w = int(image["height"]), int(image["width"])
            if row["segmentation"]["size"] != [h, w] or int(row["mask_area"]) <= 0:
                raise ValueError(f"Missing, empty or wrong-size GT mask: {key}")
            x, y = map(float, row["point_xy_px"])
            x0, y0, x1, y1 = map(float, row["bbox_xyxy_px"])
            if not (0 <= x < w and 0 <= y < h and
                    0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h and
                    int(row["point_label"]) == 1):
                raise ValueError(f"Invalid foreground point/box: {key}")

        # Official transforms handle masks, points and boxes together.
        self.transforms = [
            DecodeRle(),
            RandomResizeAPI(sizes=resolution, max_size=resolution,
                            square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5] * 3, std=[0.5] * 3),
        ]

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        from PIL import Image as PILImage
        from pycocotools import mask as mask_utils
        from sam3.train.data.sam3_image_dataset import (
            Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
        )

        row = self.rows[index]
        mode = ("point", "box")[(index + self.epoch) % 2] if self.prompt_mode == "mixed" else self.prompt_mode
        with PILImage.open(self.root / row["image_path"]) as image_file:
            image = image_file.convert("RGB")
        w, h = image.size
        mask = mask_utils.decode(row["segmentation"])
        if mask.shape != (h, w) or int(mask.sum()) != int(row["mask_area"]):
            raise ValueError("GT RLE mask/area mismatch")
        x, y = (float(v) for v in row["point_xy_px"])
        if not bool(mask[int(y), int(x)]):
            raise ValueError(f"Point outside region {row['image_id']}/{row['region_id']}")
        bbox = torch.tensor(row["bbox_xyxy_px"], dtype=torch.float32)
        target = Object(
            bbox=bbox, area=float(row["mask_area"]),
            object_id=int(row["region_id"]), frame_index=0,
            segment=row["segmentation"], source="sa1b_geometry",
        )
        # Official NormalizeAPI converts this pixel-space (x,y,label) into
        # normalized (x/w,y/h,label), while transforming the mask/box too.
        point = torch.tensor([[[x, y, 1.0]]], dtype=torch.float32)
        image_id = int(str(row["image_id"]).removeprefix("sa_"))
        query = FindQueryLoaded(
            query_text="visual",  # Matches Sam3Processor geometry-only prompts.
            image_id=0, object_ids_output=[0], is_exhaustive=False,
            is_pixel_exhaustive=False,
            input_bbox=bbox.unsqueeze(0).clone() if mode == "box" else None,
            input_bbox_label=torch.ones(1, dtype=torch.long) if mode == "box" else None,
            input_points=point if mode == "point" else None,
            inference_metadata=InferenceMetadata(
                image_id, image_id, 1, (h, w), int(row["region_id"]), 0
            ),
        )
        sample = Datapoint(
            find_queries=[query],
            images=[Image(image, [target], (h, w))],
            raw_images=[image],
        )
        for transform in self.transforms:
            sample = transform(sample, epoch=self.epoch)
        return sample


def collate_geometry(batch, dict_key="all", repeats=1, with_seg_masks=True):
    """Official Stage3 collator with 3-D point records, not PVS's 257-D default.

    A batch may contain a point-only and a box-only sample. The default
    input_points_embedding_dim=257 silently breaks such mixed collation.
    """
    from sam3.train.data.collator import collate_fn_api

    if not with_seg_masks:
        raise ValueError("Geometry distillation requires GT segmentation masks")
    return collate_fn_api(
        batch, dict_key=dict_key, repeats=repeats, with_seg_masks=True,
        input_points_embedding_dim=3,
    )
