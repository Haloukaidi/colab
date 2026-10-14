"""Real checkpoint/data smoke steps; does not establish segmentation accuracy."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-root", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--prompt-manifest", required=True)
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--rv-checkpoint", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), args.official_root,
                    str(Path(args.official_root) / "sam3")]
    import torch
    from lite50.train_data import SACapStage3Dataset, ImageFeatureDataset, TextFeatureDataset
    from sam3.train.data.collator import collate_fn_api
    from lite50.distillation import build_teacher, image_features, image_distillation, text_distillation
    from lite50.stage3 import build_student
    from lite50.train_policy import apply_train_policy, build_optimizer, parameter_report
    from lite50.train_audit import fingerprints, frozen_keys, verify_update
    from lite50.training_utils import verify_official_commit, save_feature_checkpoint, load_feature_resume
    torch.set_num_threads(2)
    torch.manual_seed(123)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    report = {"official_commit": verify_official_commit(args.official_root), "device": "cpu", "stages": []}
    def record(name, **values):
        print(json.dumps({"event": name, **values}), flush=True)
        report[name] = values
        (root / "feature_validation.json").write_text(json.dumps(report, indent=2))
    datasets = {}
    for split in ("train", "val"):
        ds = SACapStage3Dataset(args.dataset_root, args.prompt_manifest, split)
        datasets[split] = len(ds)
        for mode in ("text", "box", "text_box"):
            ds.prompt_mode = mode
            batch = collate_fn_api([ds[0]], repeats=1, dict_key="all", with_seg_masks=True)["all"]
            assert batch.find_targets[0].segments.any()
            assert tuple(batch.img_batch.shape) == (1, 3, 1008, 1008)
    record("data", approved_regions=datasets, modes=["text", "box", "text_box"])
    record("building_teacher")
    teacher = build_teacher(args.teacher_checkpoint)
    teacher_before = fingerprints(teacher)
    record("teacher", parameters=sum(p.numel() for p in teacher.parameters()),
           trainable=sum(p.numel() for p in teacher.parameters() if p.requires_grad))
    student = build_student(args.rv_checkpoint)
    record("student", **parameter_report(student))
    for phase in ("vision", "text"):
        student.zero_grad(set_to_none=True)
        apply_train_policy(student, phase)
        optimizer = build_optimizer(student, lr=1e-4)
        before, frozen = fingerprints(student), frozen_keys(student)
        start = time.monotonic()
        record("running", phase=phase)
        if phase == "vision":
            inputs = ImageFeatureDataset(args.dataset_root)[0][None]
            with torch.no_grad():
                tf, tt = image_features(teacher, inputs)
            sf, st = image_features(student, inputs)
            losses = image_distillation(sf, tf, st, tt)
        else:
            inputs = [TextFeatureDataset(args.dataset_root)[0], "a yellow bus"]
            with torch.no_grad():
                tf = teacher.backbone.forward_text(inputs, device="cpu")
            sf = student.backbone.forward_text(inputs, device="cpu")
            losses = text_distillation(sf, tf)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],
                                      1.0, error_if_nonfinite=True)
        optimizer.step()
        result = {**verify_update(student, before, frozen), "params": parameter_report(student),
                  "losses": {k: float(v.detach()) for k, v in losses.items()},
                  "seconds": time.monotonic()-start}
        checkpoint = root / phase / "checkpoint.pt"
        save_feature_checkpoint(checkpoint, student, optimizer, phase, 1, 123, args.teacher_checkpoint)
        saved = fingerprints(student)
        step, seed = load_feature_resume(checkpoint, student, optimizer, phase, args.teacher_checkpoint)
        assert saved == fingerprints(student) and step == 1 and seed == 123
        result["checkpoint_reload_exact"] = True
        report["stages"].append(result)
        record("completed_phase", phase=phase)
        del losses, tf, sf
        if phase == "vision":
            del tt, st
    assert fingerprints(teacher) == teacher_before
    assert all(p.grad is None for p in teacher.parameters())
    record("complete", teacher_frozen_unchanged=True)


if __name__ == "__main__":
    main()
