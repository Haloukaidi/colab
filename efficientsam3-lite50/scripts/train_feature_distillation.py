"""Stage1 feature objectives with Lite50 modules; use trained SAM3 as teacher."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-root", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--teacher-checkpoint", required=True)
    p.add_argument("--rv-checkpoint", required=True)
    p.add_argument("--lite-checkpoint")
    p.add_argument("--resume")
    p.add_argument("--phase", choices=("vision", "text"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=1000, help="Total optimizer steps (including resumed steps)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save-every", type=int, default=100)
    args = p.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), args.official_root,
                    str(Path(args.official_root) / "sam3")]
    import torch
    from torch.utils.data import DataLoader
    from lite50.stage3 import build_student
    from lite50.distillation import build_teacher, image_features, image_distillation, text_distillation
    from lite50.train_policy import build_optimizer, parameter_report
    from lite50.train_data import ImageFeatureDataset, TextFeatureDataset
    from lite50.training_utils import verify_official_commit, save_feature_checkpoint, load_feature_resume
    verify_official_commit(args.official_root)
    if args.steps < 1 or args.batch_size < 1 or args.save_every < 1:
        raise ValueError("steps, batch size and save interval must be positive")
    if args.phase == "text" and not (args.lite_checkpoint or args.resume):
        raise ValueError("Text phase must inherit the completed vision-phase checkpoint")
    torch.manual_seed(args.seed)
    student = build_student(args.rv_checkpoint, args.resume or args.lite_checkpoint, args.phase, args.device)
    teacher = build_teacher(args.teacher_checkpoint, args.device)
    optimizer = build_optimizer(student, args.lr)
    dataset = (ImageFeatureDataset if args.phase == "vision" else TextFeatureDataset)(args.dataset_root)
    step, seed = 0, args.seed
    if args.resume:
        step, seed = load_feature_resume(args.resume, student, optimizer, args.phase, args.teacher_checkpoint)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps(parameter_report(student)), flush=True)
    batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    while step < args.steps:
        epoch, offset = divmod(step, batches)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(seed + epoch))
        for index, batch in enumerate(loader):
            if index < offset:
                continue
            optimizer.zero_grad(set_to_none=True)
            if args.phase == "vision":
                images = batch.to(args.device)
                with torch.no_grad():
                    tf, tt = image_features(teacher, images)
                sf, st = image_features(student, images)
                losses = image_distillation(sf, tf, st, tt)
            else:
                with torch.no_grad():
                    tf = teacher.backbone.forward_text(list(batch), device=args.device)
                sf = student.backbone.forward_text(list(batch), device=args.device)
                losses = text_distillation(sf, tf)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Non-finite distillation loss")
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],
                                          1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            record = {"step": step, "phase": args.phase,
                      **{k: float(v.detach()) for k, v in losses.items()}}
            with (out / "losses.jsonl").open("a") as log:
                log.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if step % args.save_every == 0 or step == args.steps:
                save_feature_checkpoint(out / "checkpoint.pt", student, optimizer, args.phase,
                                        step, seed, args.teacher_checkpoint)
            if step == args.steps:
                break


if __name__ == "__main__":
    main()
