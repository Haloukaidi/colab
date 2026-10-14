"""Data-only adapters: processed Recap + RefCOCO text KD; original SACap GT.

No tokenizer, model, distillation objective or Stage3 trainable policy changes.
Text-only KD uses one retained manifest row per epoch. Frequency columns are
provenance, not implicit repetition/weights. Stage3 never uses short_text.
"""
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
import gzip
import hashlib
import json

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from torch.utils.data import Dataset


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_manifest(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pq.read_table(path, memory_map=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return pa.Table.from_pylist([json.loads(line) for line in stream if line.strip()])


class TextManifestDataset(Dataset):
    """Map-style text manifest compatible with the existing Stage1 resume loop."""

    def __init__(self, text_manifest, split="train"):
        path = Path(text_manifest)
        table = read_manifest(path)
        if not {"text", "split", "source"} <= set(table.column_names):
            raise ValueError("Text manifest requires text, split and source")
        self.table = table.filter(pc.equal(table["split"], split))
        if not len(self.table):
            raise ValueError(f"Empty text split: {split}")
        text = self.table["text"]
        if text.null_count or pc.any(pc.equal(pc.utf8_trim_whitespace(text), "")).as_py():
            raise ValueError("Blank/null text in KD manifest")
        self.data_provenance = {
            "manifest_sha256": file_sha256(path), "split": split,
            "rows": len(self.table), "sampling": "one_visit_per_retained_manifest_row",
        }

    def __len__(self):
        return len(self.table)

    def __getitem__(self, index):
        # Return the SAME original/curated string to teacher and student.
        # Both encoders retain the official runtime context_length=16.
        return self.table["text"][index].as_py()


def safe_child(root, relative):
    root = Path(root).resolve()
    child = (root / relative).resolve()
    if not child.is_relative_to(root):
        raise ValueError(f"Manifest path escapes dataset root: {relative}")
    return child


@lru_cache(maxsize=64)
def _original_annotations(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    annotations = data["annotations"]
    by_id = {str(a["id"]): a for a in annotations}
    if len(by_id) != len(annotations):
        raise ValueError(f"Duplicate original annotation IDs: {path}")
    return by_id


class RawSACapStage3Dataset(Dataset):
    """Original region captions + original SA-1B masks; no short-text whitelist.

    Records are grouped by image, up to32 original region queries per image,
    matching the official recipe's cap. All source rows remain in the manifest.
    Only text prompts are used. A region caption is NOT an exhaustive category
    annotation for all instances; masks must come from the linked anno_id.
    Original source captions may exceed14 BPE. Leave truncation to SimpleTokenizer.
    """

    def __init__(self, dataset_root, caption_manifest=None, native_parquet=None,
                 split="train", resolution=1008, prompt_mode="text",
                 max_segments_per_image=32, limit=None):
        if split not in ("train", "val") or prompt_mode != "text":
            raise ValueError("Official-style SACap Stage3 uses train/val and text-only prompts")
        if resolution != 1008 or max_segments_per_image < 1:
            raise ValueError("Expected resolution1008 and positive max_segments_per_image")
        self.root, self.split = Path(dataset_root), split
        self.resolution, self.max_segments = resolution, int(max_segments_per_image)
        self.epoch = 0
        caption_manifest = Path(caption_manifest or self.root / "manifests/sacap_regions.jsonl")
        native_parquet = Path(native_parquet or self.root / f"sacap/{split}.parquet")
        rows = read_manifest(caption_manifest).to_pylist()
        rows = [r for r in rows if r["split"] == split and not r.get("benchmark_excluded", False)]
        native = {}
        for batch in pq.ParquetFile(native_parquet).iter_batches(batch_size=4096):
            for image in batch.to_pylist():
                for region in image["segments_info"]:
                    key = (Path(image["imagename"]).stem, str(region["anno_id"]))
                    if key in native:
                        raise ValueError(f"Duplicate native SACap region: {key}")
                    native[key] = region["caption"]
        self.groups = OrderedDict()
        seen = set()
        for row in rows:
            key = (str(row["image_id"]), str(row["region_id"]))
            if key in seen:
                raise ValueError(f"Duplicate regional query: {key}")
            seen.add(key)
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError(f"Missing original regional caption: {key}")
            if native.get(key) != row["text"]:
                raise ValueError(f"Original SACap regional text mismatch: {key}")
            self.groups.setdefault(key[0], []).append(row)
        self.image_ids = list(self.groups)
        if limit is not None:
            if int(limit) < 1:
                raise ValueError("limit must be positive")
            self.image_ids = self.image_ids[:int(limit)]
        if not self.image_ids:
            raise ValueError("No original-caption SACap images in this split")
        self.data_provenance = {
            "caption_sha256": file_sha256(caption_manifest),
            "native_sha256": file_sha256(native_parquet), "split": split,
            "text_policy": "original_runtime_ctx16", "images": len(self.image_ids),
            "max_segments_per_image": self.max_segments,
        }
        self.transforms = None

    def __len__(self):
        return len(self.image_ids)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        import torch
        from PIL import Image as PILImage
        from pycocotools import mask as mask_utils
        from sam3.train.data.sam3_image_dataset import (
            Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata,
        )
        from sam3.train.transforms.basic_for_api import RandomResizeAPI, ToTensorAPI, NormalizeAPI
        from sam3.train.transforms.segmentation import DecodeRle

        image_id = self.image_ids[index]
        rows = self.groups[image_id][:self.max_segments]
        row = rows[0]
        image_path = safe_child(self.root, row.get("image_path") or f"sa1b/images/{self.split}/{image_id}.jpg")
        ann_path = safe_child(self.root, row.get("annotation_path") or f"sa1b/annotations/{self.split}/{image_id}.json")
        annotations = _original_annotations(str(ann_path))
        with PILImage.open(image_path) as image_file:
            image = image_file.convert("RGB")
        width, height = image.size
        objects, queries = [], []
        for row in rows:
            region_id = str(row["region_id"])
            if region_id not in annotations:
                raise ValueError(f"Missing SA-1B mask: {image_id}/{region_id}")
            segmentation = annotations[region_id]["segmentation"]
            if list(segmentation["size"]) != [height, width]:
                raise ValueError("SA-1B mask and image dimensions differ")
            if row.get("segmentation") is not None and row["segmentation"] != segmentation:
                raise ValueError("Prepared RLE differs from original SA-1B annotation")
            area = float(mask_utils.area(segmentation))
            x, y, w, h = mask_utils.toBbox(segmentation).tolist()
            if area <= 0 or not (0 <= x < x+w <= width and 0 <= y < y+h <= height):
                raise ValueError("Empty or invalid original mask")
            target_index = len(objects)
            objects.append(Object(
                bbox=torch.tensor([x, y, x+w, y+h], dtype=torch.float32),
                area=area, object_id=int(region_id), frame_index=0,
                segment=segmentation, source="sacap_original",
            ))
            numeric_id = int(image_id.removeprefix("sa_"))
            queries.append(FindQueryLoaded(
                query_text=row["text"], image_id=0, object_ids_output=[target_index],
                is_exhaustive=False, is_pixel_exhaustive=False,
                input_bbox=None, input_bbox_label=None,
                inference_metadata=InferenceMetadata(numeric_id, numeric_id, 1,
                                                      (height, width), int(region_id), 0),
            ))
        datapoint = Datapoint(find_queries=queries, images=[Image(image, objects, (height, width))], raw_images=[image])
        if self.transforms is None:
            self.transforms = [DecodeRle(), RandomResizeAPI(sizes=self.resolution,
                max_size=self.resolution, square=True, consistent_transform=False),
                ToTensorAPI(), NormalizeAPI(mean=[.5]*3, std=[.5]*3)]
        for transform in self.transforms:
            datapoint = transform(datapoint, epoch=self.epoch)
        return datapoint
