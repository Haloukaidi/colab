"""Reviewable partial adapter for official Stage3 trainer (not full trainer)."""
import argparse
from pathlib import Path
import difflib

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--upstream",default=".")
    p.add_argument("--apply",action="store_true")
    a=p.parse_args()
    target=Path(a.upstream)/"stage3/model.py"
    original=target.read_text()
    modified=original.replace(
        "from sam3.model_builder import build_efficientsam3_image_model",
        "from lite50.builder import build_lite50_image_model as build_efficientsam3_image_model",
    ).replace(
        "_set_requires_grad(model.backbone.vision_backbone.trunk, True)",
        "_set_requires_grad(model.backbone.vision_backbone.trunk, True)\n"
        "        _set_requires_grad(model.backbone.vision_backbone.convs, True)",
    ).replace(
        "model.backbone.vision_backbone.trunk.train()",
        "model.backbone.vision_backbone.trunk.train()\n"
        "        model.backbone.vision_backbone.convs.train()",
    ).replace('backbone_type="tinyvit",','backbone_type="repvit",'
    ).replace('model_name="11m",','model_name="m1.1",'
    ).replace('text_encoder_type="MobileCLIP-S1",','text_encoder_type="MobileCLIP-S0",')
    if "from lite50.builder" not in modified:
        raise RuntimeError("Upstream training file changed; cannot patch")
    print("".join(difflib.unified_diff(
        original.splitlines(keepends=True),modified.splitlines(keepends=True),
        fromfile=str(target),tofile=str(target)+".lite50")))
    if a.apply:
        target.write_text(modified)
    print("IMPORTANT: This is not a complete Stage3 integration. Adjust optimizer groups,")
    print("Hydra configuration, segmentation losses and skip_saving_parameters manually.")
if __name__=="__main__":
    main()
