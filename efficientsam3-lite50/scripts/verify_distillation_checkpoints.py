"""Strictly reload Stage3 checkpoints and check full state/optimizer coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("checkpoints", nargs="+")
    args = p.parse_args()
    sys.path[:0] = [str(Path(__file__).resolve().parents[1]), args.official_root,
                   str(Path(args.official_root) / "sam3")]
    import torch
    from lite50.stage3 import build_student
    from lite50.train_policy import parameter_report
    from lite50.training_utils import verify_official_commit
    verify_official_commit(args.official_root)
    torch.set_num_threads(2)
    results = []
    for path in args.checkpoints:
        state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        phase = state["lite50"]["phase"]
        model = build_student(None, path, phase)
        expected = {k.removeprefix("student."): v for k, v in state["model"].items()}
        actual = model.state_dict()
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[k], expected[k]) for k in actual)
        optimizer_ids = [i for g in state["optimizer"]["param_groups"] for i in g["params"]]
        assert len(optimizer_ids) == len(set(optimizer_ids))
        assert len(optimizer_ids) == sum(p.requires_grad for p in model.parameters())
        assert all(g["lr"] > 0 for g in state["optimizer"]["param_groups"])
        for entry in state["optimizer"]["state"].values():
            assert all(not isinstance(v, torch.Tensor) or torch.isfinite(v).all() for v in entry.values())
        with open(path, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        results.append({"path": str(path), "sha256": digest, "bytes": Path(path).stat().st_size,
            "strict_reload_exact": True, "optimizer_parameter_tensors": len(optimizer_ids),
            "epoch": state["epoch"], "steps": state["steps"], "parameters": parameter_report(model)})
        del model, state, actual, expected
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
