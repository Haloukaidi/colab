"""Retired partial source patch: use the complete Lite50 Stage3 adapter."""
raise SystemExit(
    "The old string-replacement adapter is retired because it unfreezes RepViT "
    "and does not fix data/optimizer integration. Use scripts/train_distill_stage3.py "
    "with configs/distill_stage3.yaml; see DISTILLATION.md. No upstream files were changed."
)
