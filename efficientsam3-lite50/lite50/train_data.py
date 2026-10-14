"""Native SACap segments_info/anno_id -> official SAM3 Stage3 Datapoint."""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image as PILImage
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x]


def image_tensor(path, resolution=1008):
    with PILImage.open(path) as im:
        im = im.convert("RGB").resize((resolution, resolution), PILImage.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 127.5 - 1


class ImageFeatureDataset(Dataset):
    def __init__(self, dataset_root, split="train", resolution=1008):
        self.root, self.resolution = Path(dataset_root), resolution
        self.rows = [r for r in read_jsonl(self.root / "manifests/images.jsonl") if r["split"] == split]
        if not self.rows:
            raise ValueError("Empty image split")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return image_tensor(self.root / self.rows[index]["image_path"], self.resolution)


class TextFeatureDataset(Dataset):
    def __init__(self, dataset_root, split="train"):
        self.rows = [r for r in read_jsonl(Path(dataset_root) / "manifests/recap_texts.jsonl")
                     if r["split"] == split]
        if not self.rows or any(not r["text"].strip() for r in self.rows):
            raise ValueError("Empty text split or blank text")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # Both teacher/student use their identical 16-context BPE truncation.
        return self.rows[index]["text"]


class SACapStage3Dataset(Dataset):
    def __init__(self, dataset_root, prompt_manifest, split="train", resolution=1008,
                 prompt_mode="mixed", include_unreviewed=False, limit=None):
        import pyarrow.parquet as pq
        from sam3.model.tokenizer_ve import SimpleTokenizer
        from sam3.train.transforms.basic_for_api import RandomResizeAPI, ToTensorAPI, NormalizeAPI
        from sam3.train.transforms.segmentation import DecodeRle
        if split not in ("train", "val"):
            raise ValueError("Use the fixed mini-v1 train/val split; no re-splitting")
        if prompt_mode not in ("text", "box", "text_box", "mixed"):
            raise ValueError("Stage3 PCS upstream ignores point prompts; use text/box/text_box")
        self.root, self.split, self.prompt_mode = Path(dataset_root), split, prompt_mode
        self.epoch = 0
        all_rows = read_jsonl(prompt_manifest)
        self.rows = [r for r in all_rows if r["split"] == split
            and r.get("review_status") not in ("hold", "rejected") and (
            (r.get("enabled_for_training") and r.get("review_status") in ("accepted", "corrected"))
            or (include_unreviewed and r.get("eligible_automatic_candidate", False)))]
        # Join native SACap parquet by image stem + numeric anno_id, not row order.
        native = {}
        for row in pq.read_table(self.root / f"sacap/{split}.parquet").to_pylist():
            for seg in row["segments_info"]:
                key = (Path(row["imagename"]).stem, str(seg["anno_id"]))
                if key in native:
                    raise ValueError(f"Duplicate SACap region: {key}")
                native[key] = seg["caption"]
        images = {r["image_id"]: r for r in read_jsonl(self.root / "manifests/images.jsonl")}
        from sam3.model import tokenizer_ve
        tokenizer = SimpleTokenizer(Path(tokenizer_ve.__file__).resolve().parents[2] / "assets/bpe_simple_vocab_16e6.txt.gz")
        seen = set()
        for row in self.rows:
            key = (str(row["image_id"]), str(row["region_id"]))
            if key in seen:
                raise ValueError(f"Duplicate prompt-sidecar region: {key}")
            seen.add(key)
            if native.get(key) != row["source_text"]:
                raise ValueError(f"SACap native annotation / prompt sidecar mismatch: {key}")
            image = images[row["image_id"]]
            if image["split"] != split or row["image_path"] != image["image_path"]:
                raise ValueError(f"Cross-split or image-path mismatch: {key}")
            phrase = row["short_text"].strip()
            if not phrase or len(tokenizer.encode(phrase)) > 14:
                raise ValueError(f"Phrase does not fit 16-token context: {key}")
            # Original SA-1B JSON remains authoritative, including exact RLE.
            ann_path = self.root / f"sa1b/annotations/{split}/{row['image_id']}.json"
            annotations = json.loads(ann_path.read_text())["annotations"]
            matched = [a for a in annotations if str(a["id"]) == str(row["region_id"])]
            if len(matched) != 1 or matched[0]["segmentation"] != row["segmentation"]:
                raise ValueError(f"Mask / anno_id mismatch: {key}")
            if mask_utils.area(row["segmentation"]) <= 0:
                raise ValueError(f"Empty mask: {key}")
        if limit is not None:
            if int(limit) < 1:
                raise ValueError("limit must be positive")
            self.rows = self.rows[:int(limit)]
        if not self.rows:
            raise ValueError("No approved SACap samples in this split")
        self.transforms = [DecodeRle(), RandomResizeAPI(sizes=resolution, max_size=resolution,
                            square=True, consistent_transform=False), ToTensorAPI(),
                           NormalizeAPI(mean=[0.5]*3, std=[0.5]*3)]

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        from sam3.train.data.sam3_image_dataset import Datapoint, Image, Object, FindQueryLoaded, InferenceMetadata
        row = self.rows[index]
        with PILImage.open(self.root / row["image_path"]) as im:
            im = im.convert("RGB")
        w, h = im.size
        if [h, w] != list(row["segmentation"]["size"]):
            raise ValueError("Image and RLE sizes differ")
        mode = ("text", "box", "text_box")[(index+self.epoch) % 3] if self.prompt_mode == "mixed" else self.prompt_mode
        bbox = torch.tensor(row["bbox_xyxy_px"], dtype=torch.float32)
        x0, y0, x1, y1 = bbox.tolist()
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("Invalid target bounding box")
        region_id, image_id = int(row["region_id"]), int(str(row["image_id"]).removeprefix("sa_"))
        obj = Object(bbox=bbox, area=float(mask_utils.area(row["segmentation"])),
                     object_id=region_id, frame_index=0, segment=row["segmentation"], source="sacap")
        q = FindQueryLoaded(query_text="visual" if mode == "box" else row["short_text"], image_id=0,
            object_ids_output=[0], is_exhaustive=False, is_pixel_exhaustive=False,
            input_bbox=bbox[None].clone() if mode != "text" else None,
            input_bbox_label=torch.ones(1, dtype=torch.long) if mode != "text" else None,
            inference_metadata=InferenceMetadata(image_id, image_id, 1, (h, w), region_id, 0))
        dp = Datapoint(find_queries=[q], images=[Image(im, [obj], (h, w))], raw_images=[im])
        for transform in self.transforms:
            dp = transform(dp, epoch=self.epoch)
        return dp
