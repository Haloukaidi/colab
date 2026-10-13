"""Compare old four-scale/slice output against the new three-scale FPN path.

Run after installing the pinned upstream EfficientSAM3 checkout and setting
PYTHONPATH to include both the Lite50 overlay and upstream sam3 package.
Uses *raw* 200-query outputs, not the possibly-empty thresholded masks.
"""
import argparse
import gc
import time
from dataclasses import replace

import torch
from PIL import Image

from lite50 import Lite50Config, build_lite50_image_model
from sam3.model.sam3_image_processor import Sam3Processor


RAW_KEYS = ("pred_masks", "pred_logits", "pred_boxes", "presence_logit_dec")


def build(weights, *, skip):
    model = build_lite50_image_model(
        lite_checkpoint_path=weights,
        config=replace(Lite50Config(), skip_unused_fpn_level=skip),
        strict_budget=False,  # measured Lite50-v0.2 is 56.41M, not <50M
        initialize_text_from_dense=False,
        device="cpu",
        eval_mode=True,
    )
    model.eval()
    return model


@torch.inference_mode()
def run_prompt(model, image, prompt, *, text, box, point):
    processor = Sam3Processor(model, device="cpu", resolution=1008, confidence_threshold=0.5)
    captured = {}
    original_forward = model.forward_grounding

    def capture(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        for key in RAW_KEYS:
            captured[key] = output[key].detach().cpu().clone()
        return output

    model.forward_grounding = capture
    try:
        state = processor.set_image(image)
        if prompt in ("text", "text_box"):
            processor.set_text_prompt(text, state)
        if prompt == "point":
            processor.add_point_prompt(point, 1, state)
        if prompt in ("box", "text_box"):
            processor.add_geometric_prompt(box, True, state)
    finally:
        model.forward_grounding = original_forward

    if set(captured) != set(RAW_KEYS):
        raise AssertionError(f"Missing raw outputs: {set(RAW_KEYS) - set(captured)}")
    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-weights", required=True, help="v0.2 student_initialized.pt")
    parser.add_argument("--image", required=True, help="Image path for identical old/new inputs")
    parser.add_argument("--text", default="person")
    parser.add_argument("--point", nargs=2, type=float, default=[400., 300.],
                        metavar=("X_PX", "Y_PX"))
    parser.add_argument("--box", nargs=4, type=float, default=[.5, .5, .5, .5],
                        metavar=("CX", "CY", "W", "H"))
    parser.add_argument("--modes", nargs="+",
                        choices=["text", "point", "box", "text_box"], default=["text"])
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    image = Image.open(args.image).convert("RGB")

    old_model = build(args.student_weights, skip=False)
    old_keys = set(old_model.state_dict())
    old_count = sum(p.numel() for p in old_model.parameters())
    before = {}
    for mode in args.modes:
        start = time.perf_counter()
        before[mode] = run_prompt(
            old_model, image, mode, text=args.text, box=args.box, point=args.point
        )
        print(f"Original FPN ({mode}) elapsed: {time.perf_counter()-start:.3f}s", flush=True)
    del old_model
    gc.collect()

    new_model = build(args.student_weights, skip=True)
    assert new_model.backbone.scalp == 0
    assert new_model.backbone.vision_backbone.active_levels == 3
    assert len(new_model.backbone.vision_backbone.convs) == 4
    assert set(new_model.state_dict()) == old_keys, "Checkpoint keys changed"
    assert sum(p.numel() for p in new_model.parameters()) == old_count
    for mode in args.modes:
        start = time.perf_counter()
        after = run_prompt(
            new_model, image, mode, text=args.text, box=args.box, point=args.point
        )
        print(f"Optimized FPN ({mode}) elapsed: {time.perf_counter()-start:.3f}s", flush=True)
        for key in RAW_KEYS:
            lhs, rhs = before[mode][key], after[key]
            if lhs.shape != rhs.shape:
                raise AssertionError((mode, key, lhs.shape, rhs.shape))
            if not torch.allclose(lhs, rhs, rtol=1e-6, atol=1e-6):
                delta = (lhs - rhs).abs().max().item()
                raise AssertionError(f"{mode}/{key} differs (max_abs_diff={delta})")
        print(f"RAW_OUTPUTS_MATCH {mode}: {', '.join(RAW_KEYS)}", flush=True)

    print(f"STRICT_WEIGHTS_COMPATIBLE params={old_count:,} skip_enabled=True", flush=True)
    print("PASS: same raw outputs / same parameter count / old v0.2 checkpoint loads")


if __name__ == "__main__":
    main()
