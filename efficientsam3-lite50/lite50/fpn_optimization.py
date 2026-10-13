"""Skip the RV-M FPN scale discarded by SAM3VLBackbone(scalp=1).

The fourth FPN module is kept in `convs` for parameter / state_dict
compatibility with Lite50-v0.2 checkpoints. Its forward pass, including
positional encoding, is skipped. No trained weights are changed.
"""
from torch import nn


class ActiveScaleFPN(nn.Module):
    """Return only the FPN levels consumed by downstream SAM3 modules."""

    def __init__(self, neck: nn.Module, active_levels: int):
        super().__init__()
        if not 1 <= active_levels <= len(neck.convs):
            raise ValueError("Invalid number of active FPN levels")
        if getattr(neck, "sam2_convs", None) is not None:
            raise ValueError("Optional SAM2 tracking neck is unsupported")

        # Keep exactly the upstream module names and registrations so older
        # Lite50-v0.2 state_dicts still load with strict=True.
        self.trunk = neck.trunk
        self.position_encoding = neck.position_encoding
        self.convs = neck.convs
        self.scale_factors = neck.scale_factors
        self.sam2_convs = None
        self.active_levels = active_levels

    def forward(self, tensor_list):
        x = self.trunk(tensor_list)[-1]
        features, positions = [], []
        for index in range(self.active_levels):
            feat = self.convs[index](x)
            features.append(feat)
            positions.append(self.position_encoding(feat).to(feat.dtype))
        return features, positions, None, None


def enable_skip_unused_fpn(backbone: nn.Module) -> None:
    """Move scalp=1 truncation before FPN execution without changing output.

    EfficientSAM3 RV-M currently builds four levels [4,2,1,0.5].
    The image-only VL backbone uses scalp=1, discarding level 3 AFTER
    generating it. This function makes the neck compute only levels 0..2
    and resets the downstream scalp to 0 to avoid double truncation.
    """
    neck = backbone.vision_backbone
    if (
        getattr(backbone, "scalp", None) != 1
        or len(neck.convs) != 4
        or tuple(neck.scale_factors) != (4.0, 2.0, 1.0, 0.5)
    ):
        raise ValueError("Expected RV-M four-scale FPN with scalp=1")

    backbone.vision_backbone = ActiveScaleFPN(
        neck, active_levels=len(neck.convs) - backbone.scalp
    )
    backbone.scalp = 0
