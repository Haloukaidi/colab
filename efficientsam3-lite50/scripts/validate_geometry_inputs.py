"""Fast real-SA-1B point/box preflight, without loading a 840M teacher.

Check this before the costly two-step Stage3 geometry smoke. The full
student/teacher forward and optimizer are validated by train_distill_stage3.py.
"""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()
    sys.path[:0] = [
        str(Path(__file__).resolve().parents[1]),
        str(args.official_root),
        str(Path(args.official_root) / "sam3"),
    ]
    from lite50.training_utils import verify_official_commit
    verify_official_commit(args.official_root)
    import torch
    from lite50.geometry_data import SA1BGeometryStage3Dataset, collate_geometry
    from lite50.distillation import geometry_prompt_from_find_stage

    for split in ("train", "val"):
        ds = SA1BGeometryStage3Dataset(args.dataset_root, split=split, prompt_mode="mixed")
        for mode in ("point", "box"):
            ds.prompt_mode = mode
            dp = ds[0]
            query = dp.find_queries[0]
            assert query.query_text == "visual"
            assert bool(dp.images[0].objects[0].segment.any())
            inp = collate_geometry([dp])["all"].find_inputs[0]
            prompt = geometry_prompt_from_find_stage(inp, require_geometry=True)
            assert tuple(dp.images[0].data.shape) == (3, 1008, 1008)
            if mode == "point":
                assert prompt.point_embeddings.shape == (1, 1, 2)
                assert prompt.point_labels.tolist() == [[1]]
                assert prompt.box_embeddings.shape[0] == 0
                assert ((prompt.point_embeddings >= 0) & (prompt.point_embeddings <= 1)).all()
            else:
                assert prompt.box_embeddings.shape == (1, 1, 4)
                assert prompt.point_embeddings.shape[0] == 0
            print(f"PASS {split}/{mode}: normalized geometry and GT mask", flush=True)

        # Most important official-collator regression: point-only and box-only
        # can share a batch without mixing up the point/box masks.
        ds.prompt_mode = "mixed"
        ds.set_epoch(0)
        batch = collate_geometry([ds[0], ds[1]])["all"]
        stage = batch.find_inputs[0]
        prompt = geometry_prompt_from_find_stage(stage, require_geometry=True)
        assert stage.input_points.shape == (2, 1, 3)
        assert stage.input_boxes.shape == (1, 2, 4)
        assert not bool(prompt.point_mask[0, 0]) and bool(prompt.point_mask[1, 0])
        assert bool(prompt.box_mask[0, 0]) and not bool(prompt.box_mask[1, 0])
        assert torch.count_nonzero(batch.find_targets[0].segments).item() > 0
        print(f"PASS {split}/mixed: point + box share batch, each has its own GT mask", flush=True)

    print("GEOMETRY_INPUT_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__":
    main()
