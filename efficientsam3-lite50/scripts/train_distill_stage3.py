"""Launch upstream Stage3 Trainer: official-style SACap GT baseline by default."""
import argparse
import os
from pathlib import Path
import socket
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-root", required=True)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/distill_stage3.yaml"))
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), args.official_root,
                    str(Path(args.official_root) / "sam3")]
    from lite50.training_utils import verify_official_commit
    verify_official_commit(args.official_root)
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    import torch
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    # Fail before loading any model: the supervised recipe must NEVER
    # instantiate the full SAM3 teacher, even if someone copied KD settings.
    builder = str(cfg.trainer.model._target_)
    loss = str(cfg.trainer.loss.all._target_)
    if builder == "lite50.stage3.build_stage3_supervised":
        if loss != "lite50.stage3.SupervisedLoss":
            raise ValueError("Supervised Stage3 requires SupervisedLoss (no KD terms)")
        if any(name in cfg.trainer.model for name in (
            "teacher_checkpoint", "teacher_cache_dir",
            "image_weight", "text_weight", "mask_kd_weight",
        )):
            raise ValueError("Supervised Fusion config must not declare teacher/KD settings")
        if cfg.phase not in ("stage3", "fusion1", "fusion2", "fusion4"):
            raise ValueError("Supervised Stage3 must be stage3 or an explicit Fusion ablation")
    elif builder == "lite50.stage3.build_stage3_distiller":
        if loss != "lite50.stage3.DistillationLoss":
            raise ValueError("Teacher KD requires DistillationLoss")
        if not cfg.trainer.model.get("teacher_checkpoint"):
            raise ValueError("Teacher KD needs a trained SAM3 checkpoint")
    else:
        raise ValueError(f"Unsupported Lite50 Stage3 model builder: {builder}")
    if not cfg.paths.lite_checkpoint:
        raise ValueError("Set paths.lite_checkpoint to the previous curriculum phase checkpoint")
    if cfg.resolution != 1008:
        raise ValueError("The RV-M projection outputs a fixed 72x72 grid; use resolution=1008")
    if cfg.accelerator == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Explicitly set accelerator=cpu for a CPU smoke test")
    if cfg.accelerator == "cpu" and cfg.trainer.optim.amp.enabled:
        raise ValueError("The upstream AMP loop targets CUDA; disable AMP for CPU validation")
    if "RANK" not in os.environ:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE="1",
                          MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    output = Path(cfg.paths.output)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output / "resolved_config.yaml", resolve=True)
    # Fail on missing training-only dependencies before loading an 840M teacher.
    instantiate(cfg.trainer.loss, _convert_="all")
    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


if __name__ == "__main__":
    main()
