"""Report full Lite50 parameters; the old 50M limit is optional."""
import argparse

from lite50 import Lite50Config, build_lite50_image_model, num_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Original RV-M weights, optional")
    parser.add_argument("--lite-checkpoint", help="Initialized/trained Lite50 weights")
    parser.add_argument("--no-svd", action="store_true")
    parser.add_argument(
        "--strict-budget", action="store_true",
        help="Raise an error if above Lite50Config.parameter_budget (legacy 50M target)",
    )
    args = parser.parse_args()
    model = build_lite50_image_model(
        checkpoint_path=args.checkpoint,
        lite_checkpoint_path=args.lite_checkpoint,
        initialize_text_from_dense=bool(args.checkpoint) and not args.no_svd,
        strict_budget=args.strict_budget,
    )
    vision = model.backbone.vision_backbone
    parts = {
        "RepViT backbone": vision.trunk.model.backbone,
        "Projection": vision.trunk.model.head,
        "FPN (all registered)": vision.convs,
        "Text": model.backbone.language_backbone,
        "Fusion encoder": model.transformer.encoder,
        "Transformer decoder": model.transformer.decoder,
        "Geometry": model.geometry_encoder,
        "Segmentation": model.segmentation_head,
        "Scoring": model.dot_prod_scoring,
    }
    for name, part in parts.items():
        print(f"{name:24s}: {num_params(part):>12,}")

    params = num_params(model)
    print("Original parameters:", f"{model.lite50_original_params:,}")
    print("Lite50 parameters:", f"{params:,}")
    print("Active FPN levels:", getattr(vision, "active_levels", len(vision.convs)))
    print("Registered FPN levels:", len(vision.convs))
    print("VL scalp (0 means already trimmed):", model.backbone.scalp)
    print(
        "Legacy budget <", f"{Lite50Config().parameter_budget:,}:",
        params < Lite50Config().parameter_budget,
    )


if __name__ == "__main__":
    main()
