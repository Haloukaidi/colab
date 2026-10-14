"""Thin adaptation of the actual upstream Stage3 Trainer/optimizer/loss stack."""
import torch
from torch import nn
import contextlib
from unittest.mock import patch
import sam3.train.trainer as upstream_trainer
from sam3.train.trainer import Trainer, unwrap_ddp_if_wrapped
from sam3.train.optim.optimizer import construct_optimizer
from .train_policy import apply_train_policy, assert_optimizer_coverage
from .distillation import build_teacher, Lite50Distiller, OFFICIAL_COMMIT


class CPURSSMeter(upstream_trainer.MemMeter):
    """CPU adapter for the upstream GPU-only memory meter (process RSS, GB)."""
    def update(self, n=1, reset_peak_usage=True):
        import psutil
        self.val = psutil.Process().memory_info().rss / 1e9
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
        self.peak = max(self.peak, self.val)


def build_student(rv_checkpoint, lite_checkpoint=None, phase="vision", device="cpu"):
    from sam3.model_builder import build_efficientsam3_image_model
    from .builder import make_lite50
    model = build_efficientsam3_image_model(device="cpu", eval_mode=False, load_from_HF=False,
        backbone_type="repvit", model_name="m1.1", text_encoder_type="MobileCLIP-S0",
        text_encoder_context_length=16, enable_inst_interactivity=False, enable_segmentation=True)
    if not rv_checkpoint and not lite_checkpoint:
        raise ValueError("Require RV-M initialization or a full Lite50 checkpoint")
    if rv_checkpoint and not lite_checkpoint:
        state = torch.load(rv_checkpoint, map_location="cpu", weights_only=True, mmap=True)
        clean = {k.removeprefix("detector.").replace("student_trunk.", ""): v
                 for k, v in state.get("model", state).items()}
        # Upstream loads the 77-position MobileCLIP table then truncates it to 16.
        for key, value in model.state_dict().items():
            if key in clean and clean[key].shape != value.shape and "pos_embed" in key:
                if clean[key].ndim == value.ndim and clean[key].shape[-1] == value.shape[-1]:
                    clean[key] = clean[key][..., :value.shape[-2], :]
        model.load_state_dict(clean, strict=True)
    make_lite50(model, initialize_text_from_dense=not bool(lite_checkpoint))
    if lite_checkpoint:
        state = torch.load(lite_checkpoint, map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict({k.removeprefix("student."): v
                               for k, v in state.get("model", state).items()}, strict=True)
    apply_train_policy(model, phase)
    return model.to(device)


def build_stage3_distiller(rv_checkpoint, teacher_checkpoint, lite_checkpoint=None,
                          phase="fusion1", device="cpu", **loss_weights):
    if phase not in ("geometry", "fusion1", "fusion2", "fusion4"):
        raise ValueError("Stage3 phases are geometry/fusion1/fusion2/fusion4")
    student = build_student(rv_checkpoint, lite_checkpoint, phase, device)
    teacher = build_teacher(teacher_checkpoint, device)
    model = Lite50Distiller(student, teacher, **loss_weights)
    model.provenance = {"official_commit": OFFICIAL_COMMIT, "phase": phase,
                        "teacher_checkpoint": str(teacher_checkpoint), "context_length": 16}
    return model


class Lite50SupervisedModel(nn.Module):
    """Student-only SAM3 PCS Stage3: no teacher, no feature/mask KD.

    Keep the 'student.' state_dict prefix so the existing full checkpoint
    importer, phase transitions and inference export remain compatible.
    Official SAM3Image.forward performs GT Hungarian matching during training
    and returns SAM3Output for the original Stage3 supervised loss.
    SACap Stage3 uses text/box/text+box; point grounding is trained separately
    by the preceding SA-1B geometry-KD phase.
    """

    def __init__(self, student, phase):
        super().__init__()
        self.student = student
        self.provenance = {
            "official_commit": OFFICIAL_COMMIT,
            "phase": phase,
            "objective": "supervised",
            "context_length": 16,
        }

    def forward(self, batch):
        return self.student(batch)

    def back_convert(self, targets):
        return self.student.back_convert(targets)


def build_stage3_supervised(rv_checkpoint, lite_checkpoint=None,
                            phase="fusion1", device="cpu"):
    """Default Fusion recipe, matching EfficientSAM3's GT-supervised Stage3."""
    if phase not in ("fusion1", "fusion2", "fusion4"):
        raise ValueError("Supervised Stage3 phases are fusion1/fusion2/fusion4")
    if not lite_checkpoint:
        raise ValueError(
            "Supervised Stage3 requires a Lite50 checkpoint from the preceding "
            "text/geometry/fusion phase"
        )
    student = build_student(rv_checkpoint, lite_checkpoint, phase, device)
    return Lite50SupervisedModel(student, phase)


class SupervisedLoss(nn.Module):
    """Upstream SAM3 Stage3 box/class/presence/focal-mask/Dice loss only.

    On CPU, use the *official* PyTorch focal implementation rather than its
    default GPU-only Triton kernel. On CUDA leave official behavior unchanged.
    """

    def __init__(self, supervised):
        super().__init__()
        self.supervised = supervised

    def _compute_supervised(self, outputs, targets):
        from sam3.train.loss import loss_fns
        first = next(iter(outputs))
        original = loss_fns.sigmoid_focal_loss

        def cpu_focal(*args, **kwargs):
            kwargs["triton"] = False
            return original(*args, **kwargs)

        context = (patch.object(loss_fns, "sigmoid_focal_loss", cpu_focal)
                   if first["pred_masks"].device.type == "cpu"
                   else contextlib.nullcontext())
        with context:
            return self.supervised(outputs, targets)

    def forward(self, outputs, targets):
        if any("lite50_kd" in out for out in outputs):
            raise ValueError("SupervisedLoss received KD predictions; use DistillationLoss")
        return self._compute_supervised(outputs, targets)


class DistillationLoss(SupervisedLoss):
    """Optional KD ablation, retained for explicit experiments and geometry."""

    def forward(self, outputs, targets):
        losses = self._compute_supervised(outputs, targets)
        kd = [out["lite50_kd"] for out in outputs]
        losses["core_loss"] = losses["core_loss"] + sum(x["total"] for x in kd)
        for name in ("image_kd", "text_kd", "geometry_mask_kd", "teacher_mask_pairs"):
            losses[name] = sum(x[name].detach() for x in kd)
        return losses

class Lite50Trainer(Trainer):
    def __init__(self, *, audit_updates=False, **kwargs):
        self.audit_updates = audit_updates
        # Upstream logs the entire process environment, which is unnecessary
        # for reproducibility and can contain service credentials.
        with patch.object(upstream_trainer, "log_env_variables", lambda: None):
            super().__init__(**kwargs)

    def _construct_optimizers(self):
        allow = {n for n, p in self.model.named_parameters() if p.requires_grad}
        # The upstream validator expects ALL params, including frozen ones.
        # Supply an explicit allowlist and validate exact TRAINABLE coverage here.
        self.optim = construct_optimizer(self.model, self.optim_conf.optimizer,
            self.optim_conf.options, self.optim_conf.param_group_modifiers,
            param_allowlist=allow, validate_param_groups=False)
        assert_optimizer_coverage(self.model, self.optim.optimizer)
        if any(g["lr"] <= 0 for g in self.optim.optimizer.param_groups):
            raise RuntimeError("A trainable optimizer group has zero learning rate")

    def train_epoch(self, train_loader):
        context = (patch.object(upstream_trainer, "MemMeter", CPURSSMeter)
                   if self.device.type == "cpu" else contextlib.nullcontext())
        with context:
            return self._train_epoch_with_audit(train_loader)

    def _train_epoch_with_audit(self, train_loader):
        if not self.audit_updates:
            return super().train_epoch(train_loader)
        import json
        from pathlib import Path
        from .train_audit import fingerprints, frozen_keys, verify_update
        model = unwrap_ddp_if_wrapped(self.model)
        before, frozen = fingerprints(model.student), frozen_keys(model.student)
        teacher = getattr(model, "teacher", None)
        teacher_before = fingerprints(teacher) if teacher is not None else None
        result = super().train_epoch(train_loader)
        audit = verify_update(model.student, before, frozen)
        if teacher is not None:
            if fingerprints(teacher) != teacher_before or any(p.grad is not None for p in teacher.parameters()):
                raise AssertionError("Teacher weights/buffers/gradients changed")
            audit["teacher_frozen_unchanged"] = True
        else:
            audit["teacher_forward_disabled"] = True
        audit["losses"] = {k: float(v) for k, v in result.items() if k.startswith("Losses/")}
        audit["learning_rates"] = [g["lr"] for g in self.optim.optimizer.param_groups]
        teacher_cache = getattr(model, "teacher_cache", None)
        if teacher_cache:
            audit["teacher_cache"] = {"hits": teacher_cache.hits, "misses": teacher_cache.misses}
        Path(self.logging_conf.log_dir, "update_audit.json").write_text(json.dumps(audit, indent=2))
        return result

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        checkpoint["lite50"] = unwrap_ddp_if_wrapped(self.model).provenance
        super()._save_checkpoint(checkpoint, checkpoint_path)

    def _load_resuming_checkpoint(self, ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        current = unwrap_ddp_if_wrapped(self.model).provenance
        if state.get("lite50") != current:
            raise ValueError("Resume requires matching objective/phase/teacher/context/commit. "
                             "For a new phase or objective, use lite_checkpoint with a fresh output directory.")
        del state
        super()._load_resuming_checkpoint(ckpt_path)
