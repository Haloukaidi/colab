"""SA-1B image features and legacy raw-text reading.

Current text/Stage3 readers live in lite50.data_v2. The retired SACap
short-phrase whitelist reader was removed; original captions are authoritative.
"""
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image as PILImage
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

