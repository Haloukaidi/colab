"""Reproducibility and atomic, full-student checkpoint helpers."""
from pathlib import Path
import subprocess
import torch
from .distillation import OFFICIAL_COMMIT


def verify_official_commit(root):
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"Expected official {OFFICIAL_COMMIT}, got {commit}")
    return commit


def save_feature_checkpoint(path, model, optimizer, phase, step, seed, teacher_path, data_provenance=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "phase": phase,
            "step": step, "seed": seed, "official_commit": OFFICIAL_COMMIT,
            "teacher_checkpoint": str(teacher_path), "context_length": 16,
            "data_provenance": data_provenance,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    temp = path.with_suffix(".tmp")
    torch.save(data, temp)
    temp.replace(path)


def load_feature_resume(path, model, optimizer, phase, teacher_path, data_provenance=None):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state["phase"] != phase or state["official_commit"] != OFFICIAL_COMMIT
            or state["teacher_checkpoint"] != str(teacher_path) or state["context_length"] != 16):
        raise ValueError("Resume phase/teacher/context/commit mismatch; use --lite-checkpoint for phase transition")
    if state.get("data_provenance") != data_provenance:
        raise ValueError("Text data manifest changed; start a new phase via --lite-checkpoint")
    model.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"], state["seed"]
