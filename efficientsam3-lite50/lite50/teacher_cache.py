"""Opt-in exact-input teacher cache. No cross-image/text/geometry reuse."""
import hashlib
import json
import os
from pathlib import Path
import torch
from torch.utils._pytree import tree_map_only


def update_tensor_hash(digest, value):
    if value is None:
        digest.update(b"None")
        return
    value = value.detach().cpu().contiguous()
    digest.update(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())


def model_signature(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        update_tensor_hash(digest, value)
    return digest.hexdigest()


def cache_key(teacher_signature, batch):
    from .distillation import OFFICIAL_COMMIT
    digest = hashlib.sha256(json.dumps({"schema": 1, "teacher": teacher_signature,
        "official_commit": OFFICIAL_COMMIT, "context_length": 16,
        "texts": list(batch.find_text_batch), "torch": str(torch.__version__),
        "device_type": batch.img_batch.device.type,
        "autocast_enabled": torch.is_autocast_enabled(batch.img_batch.device.type),
        "autocast_dtype": str(torch.get_autocast_dtype(batch.img_batch.device.type))},
        sort_keys=True).encode())
    update_tensor_hash(digest, batch.img_batch)
    inp = batch.find_inputs[0]
    for name in ("img_ids", "text_ids", "input_boxes", "input_boxes_mask", "input_boxes_label"):
        digest.update(name.encode())
        update_tensor_hash(digest, getattr(inp, name))
    return digest.hexdigest()


class TeacherCache:
    def __init__(self, directory):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.signature = None
        self.hits = self.misses = 0

    def get_or_compute(self, teacher, batch, compute):
        if self.signature is None:
            self.signature = model_signature(teacher)
        key = cache_key(self.signature, batch)
        path = self.root / (key + ".pt")
        if path.exists():
            record = torch.load(path, map_location="cpu", weights_only=True)
            if record["key"] != key:
                raise ValueError("Teacher cache identity mismatch")
            self.hits += 1
            return tree_map_only(torch.Tensor, lambda x: x.to(batch.img_batch.device), record["result"])
        result = compute()
        cpu_result = tree_map_only(torch.Tensor, lambda x: x.detach().cpu(), result)
        temp = path.with_suffix(f".tmp.{os.getpid()}")
        torch.save({"key": key, "result": cpu_result}, temp)
        temp.replace(path)
        self.misses += 1
        return result
