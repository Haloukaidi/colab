"""Launch the upstream Stage3 Trainer with an external, complete Hydra config."""
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
