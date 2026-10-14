"""SAM3 -> Lite50 feature and mask distillation, pinned to upstream bd0936c.

Stage1 loss formulas below are adapted verbatim (formatting aside) from
stage1/train_{image,text}_encoder_stage1.py. Image MSE sums CHANNELS,
whereas text MSE averages channels; keep that difference and weight explicitly.
Geometry mask losses are imported directly from upstream at runtime.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

OFFICIAL_COMMIT = "bd0936c788fed8d51fa799437f05abd97b401b06"


def build_teacher(checkpoint_path, device="cpu", bpe_path=None):
    from sam3.model_builder import build_sam3_image_model
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise FileNotFoundError("A local, trained SAM3 checkpoint is required")
    teacher = build_sam3_image_model(device="cpu", eval_mode=False,
                                    load_from_HF=False, bpe_path=bpe_path)
    # mmap avoids a second resident 3.45 GB copy. No permissive partial load:
    # ignore only unrelated tracker entries, require every image-model tensor.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    ckpt = ckpt.get("model", ckpt)
    clean = {k.removeprefix("detector."): v for k, v in ckpt.items()}
    keys = teacher.state_dict().keys()
    # Keep frozen CPU weights file-backed so the OS can reclaim their pages
    # during student backward. Copying into constructor tensors leaves an
    # unnecessary ~3.36 GB of anonymous RAM resident alongside activations.
    # On CUDA, .to(device) below materializes the usual device weights.
    teacher.load_state_dict({k: clean[k] for k in keys if k in clean}, strict=True, assign=True)
    teacher.backbone.language_backbone.context_length = 16
    teacher.requires_grad_(False).eval().to(device)
    # dict_keys keeps the old state-dict tensors alive until explicitly released.
    # Some CPU allocators retain those constructor allocations after replacement.
    del keys, clean, ckpt
    if torch.device(device).type == "cpu":
        import gc
        import ctypes
        gc.collect()
        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if trim is not None:  # glibc only; other platforms safely skip this hint.
            trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
            trim(0)
    return teacher


def image_features(model, images):
    captured = []
    hook = model.backbone.vision_backbone.trunk.register_forward_hook(
        lambda _m, _i, out: captured.append(out[-1]))
    try:
        features = model.backbone.forward_image(images)
    finally:
        hook.remove()
    return features, captured[0]


def image_mse(preds, teacher, mask):
    diff = (preds - teacher) * mask
    denom = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    return (diff.square().sum(dim=(1, 2, 3)) / denom).mean()


def image_cosine(preds, teacher, mask):
    sim = F.cosine_similarity(preds, teacher, dim=1)
    mask = mask.squeeze(1)
    return (((1.0 - sim) * mask).sum(dim=(1, 2)) /
            mask.sum(dim=(1, 2)).clamp(min=1.0)).mean()


def text_mse(preds, teacher, valid):
    diff = (preds - teacher) * valid.unsqueeze(-1)
    denom = valid.sum(dim=1).clamp(min=1.0) * preds.size(2)
    return (diff.square().sum(dim=(1, 2)) / denom).mean()


def text_cosine(preds, teacher, valid):
    loss = (1.0 - F.cosine_similarity(preds, teacher, dim=2)) * valid
    return (loss.sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)).mean()


def image_distillation(student, teacher, student_trunk, teacher_trunk,
                       trunk_weight=0.0015, fpn_weight=0.1, cosine_weight=0.1):
    if student_trunk.shape != teacher_trunk.shape:
        raise ValueError(f"Trunk feature mismatch: {student_trunk.shape}, {teacher_trunk.shape}")
    mask = torch.ones_like(student_trunk[:, :1])  # square warp, no padded pixels
    trunk = image_mse(student_trunk.float(), teacher_trunk.float(), mask.float())
    trunk_cos = image_cosine(student_trunk.float(), teacher_trunk.float(), mask.float())
    sf, tf = student["backbone_fpn"], teacher["backbone_fpn"]
    if len(sf) != len(tf):
        raise ValueError("Teacher/student must expose the same active FPN scales")
    fpn = []
    for s, t in zip(sf, tf):
        if s.shape != t.shape:
            raise ValueError(f"FPN feature mismatch: {s.shape}, {t.shape}")
        fpn.append(F.mse_loss(s.float(), t.float()) + cosine_weight *
                   image_cosine(s.float(), t.float(), torch.ones_like(s[:, :1]).float()))
    fpn = torch.stack(fpn).mean()
    return {"trunk_mse": trunk, "trunk_cosine": trunk_cos, "fpn_kd": fpn,
            "loss": trunk_weight * trunk + cosine_weight * trunk_cos + fpn_weight * fpn}


def text_distillation(student, teacher, cosine_weight=0.1):
    if not torch.equal(student["language_mask"], teacher["language_mask"]):
        raise ValueError("Teacher/student token masks differ; check tokenizer/context alignment")
    s = student["language_features"].transpose(0, 1).float()
    t = teacher["language_features"].transpose(0, 1).float()
    if s.shape != t.shape or s.shape[1] != 16:
        raise ValueError(f"Expected aligned 16-token features: {s.shape}, {t.shape}")
    valid = (~student["language_mask"]).float()
    mse, cosine = text_mse(s, t, valid), text_cosine(s, t, valid)
    return {"text_mse": mse, "text_cosine": cosine, "loss": mse + cosine_weight * cosine}


def matching_by_target(indices):
    # Upstream matcher returns packed (batch_idx, query_idx, target_idx).
    batch, query, target = indices
    if target is None:
        target = torch.arange(len(query), device=query.device)
    return {(int(b), int(t)): int(q) for b, q, t in zip(batch, query, target)}


def matched_geometry_kd(student_out, teacher_out, targets, min_teacher_iou=0.3):
    """Match independently to the SAME GT region, never assume query order.

    Only teacher masks that overlap the known region are soft targets. The
    official Stage3 supervised mask loss remains active for every GT region.
    """
    from stage1_geometry_finetune.losses import GeometryFinetuningLoss
    si, ti = matching_by_target(student_out["indices"]), matching_by_target(teacher_out["indices"])
    students, teachers = [], []
    for key in sorted(si.keys() & ti.keys()):
        b, target_idx = key
        sm = student_out["pred_masks"][b, si[key]]
        tm = teacher_out["pred_masks"][b, ti[key]].detach()
        gt = targets["masks"][target_idx].float()
        gt = F.interpolate(gt[None, None], size=tm.shape[-2:], mode="nearest")[0, 0] > 0.5
        pred = tm > 0
        iou = (pred & gt).sum().float() / (pred | gt).sum().clamp(min=1)
        if iou.item() >= min_teacher_iou:
            students.append(sm)
            teachers.append(tm)
    if not students:
        return student_out["pred_masks"].sum() * 0.0, 0
    result = GeometryFinetuningLoss(embedding_weight=0)(
        student_masks=torch.stack(students).float(), teacher_masks=torch.stack(teachers).float())
    return result["total_loss"], len(students)


class Lite50Distiller(nn.Module):
    """Stage3-compatible student; the immutable teacher is not checkpointed/DDP-reduced."""
    def __init__(self, student, teacher, image_weight=1.0, text_weight=1.0,
                 mask_kd_weight=0.5, min_teacher_iou=0.3, teacher_cache_dir=None):
        super().__init__()
        self.student = student
        object.__setattr__(self, "teacher", teacher)
        self.image_weight, self.text_weight = image_weight, text_weight
        self.mask_kd_weight, self.min_teacher_iou = mask_kd_weight, min_teacher_iou
        from .teacher_cache import TeacherCache
        self.teacher_cache = TeacherCache(teacher_cache_dir) if teacher_cache_dir else None

    def back_convert(self, target):
        return self.student.back_convert(target)

    def forward(self, batch):
        from sam3.model.geometry_encoders import Prompt
        from sam3.model.model_misc import SAM3Output
        teacher = self.teacher.to(batch.img_batch.device).eval()
        def compute_teacher():
            tv, tt = image_features(teacher, batch.img_batch)
            tl = teacher.backbone.forward_text(batch.find_text_batch, device=batch.img_batch.device)
            inp = batch.find_inputs[0]
            tout = teacher.forward_grounding({**tv, **tl}, inp, None,
                Prompt(box_embeddings=inp.input_boxes, box_mask=inp.input_boxes_mask,
                       box_labels=inp.input_boxes_label))
            # Avoid retaining encoder intermediates in the cache.
            tout = {k: tout[k] for k in ("pred_masks", "pred_logits", "pred_boxes", "pred_boxes_xyxy")}
            return tv, tt, tl, tout
        with torch.no_grad():
            tv, tt, tl, tout = (self.teacher_cache.get_or_compute(teacher, batch, compute_teacher)
                               if self.teacher_cache else compute_teacher())
            target = self.back_convert(batch.find_targets[0])
            teacher._compute_matching(tout, target)
        sv, st = image_features(self.student, batch.img_batch)
        sl = self.student.backbone.forward_text(batch.find_text_batch, device=batch.img_batch.device)
        inp = batch.find_inputs[0]
        out = self.student.forward_grounding({**sv, **sl}, inp, batch.find_targets[0],
            Prompt(box_embeddings=inp.input_boxes, box_mask=inp.input_boxes_mask,
                   box_labels=inp.input_boxes_label))
        # Evaluation-mode student forward does not compute matching itself.
        if "indices" not in out:
            self.student._compute_matching(out, target)
        iv = image_distillation(sv, tv, st, tt)
        tx = text_distillation(sl, tl)
        mk, count = matched_geometry_kd(out, tout, target, self.min_teacher_iou)
        out["lite50_kd"] = {"image_kd": iv["loss"], "text_kd": tx["loss"],
                            "geometry_mask_kd": mk,
                            "teacher_mask_pairs": mk.new_tensor(float(count)),
                            "total": self.image_weight * iv["loss"] + self.text_weight * tx["loss"]
                                     + self.mask_kd_weight * mk}
        outputs = SAM3Output(iter_mode=SAM3Output.IterMode.LAST_STEP_PER_STAGE)
        outputs.append([out])
        return outputs
