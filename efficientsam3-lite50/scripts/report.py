"""Print model parameter breakdown and assert strict <50M budget."""
import argparse
from lite50 import build_lite50_image_model, num_params
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--lite-checkpoint")
    p.add_argument("--no-svd", action="store_true")
    a = p.parse_args()
    model = build_lite50_image_model(
        checkpoint_path=a.checkpoint, lite_checkpoint_path=a.lite_checkpoint,
        initialize_text_from_dense=bool(a.checkpoint) and not a.no_svd)
    v = model.backbone.vision_backbone
    parts = {
        "RepViT backbone": v.trunk.model.backbone,
        "Projection": v.trunk.model.head,
        "FPN": v.convs,
        "Text": model.backbone.language_backbone,
        "Fusion encoder": model.transformer.encoder,
        "Transformer decoder": model.transformer.decoder,
        "Geometry": model.geometry_encoder,
        "Segmentation": model.segmentation_head,
        "Scoring": model.dot_prod_scoring,
    }
    for name, part in parts.items():
        print(f"{name:24s}: {num_params(part):>12,}")
    print("Original parameters:", f"{model.lite50_original_params:,}")
    print("Lite50 parameters:", f"{num_params(model):,}")
    assert num_params(model) < 50_000_000
if __name__ == "__main__":
    main()
