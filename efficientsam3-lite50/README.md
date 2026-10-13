# EfficientSAM3 RV-M Lite50 — experimental code overlay

A **research prototype (not a trained checkpoint)** based on [EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3). Designed to **target <50M total parameters** while preserving SAM3 text/concept, box and point prompt interfaces. The complete model parameter count and segmentation quality **must be validated on the installed upstream version**.

## Architecture changes

| Part | RV-M | Lite50 design |
|---|---|---|
| Backbone | RepViT-M1.1 | keep |
| Projection | 512→1024→dense 3×3 1024 | 512→256→DW3×3→1024 |
| FPN | Transposed convolution + four dense 3×3 convs | projection at low res, bilinear up/downsampling + depthwise-pointwise convs |
| Text tokenizer | MobileCLIP-S0 vocabulary (49,408 entries) | unchanged |
| Text embedding | 49,408×512 | rank-64 factorization with optional SVD initialization |
| MobileCLIP-S0 attention | 4 transformer layers | 2, keeping first and last |
| Vision-language fusion encoder | 6 layers | 4 |
| Geometry encoder | 3 layers | 2 (point/box path retained) |
| Decoder, segmentation, scoring | original | unchanged |

The modified Projection, FPN and factorized embedding together **save exactly 38,147,072 parameters** according to standalone PyTorch definitions matching the original layers; layer pruning saves additional parameters. **That number is not a measurement of the full model**, so use `scripts/report.py` for authoritative total and a strict <50M assertion.

## Setup

Clone [EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3), install official requirements, and copy the files in this directory (`lite50/`, `scripts/`, `tests/`) into the checkout. Set `PYTHONPATH="$PWD:$PWD/sam3:$PYTHONPATH"`, then:

```bash
python -m unittest discover -s tests -v
python scripts/report.py --no-svd
# or load the original RV-M model before structural replacements:
python scripts/report.py --checkpoint /path/to/efficientsam3_repvit.pt
```

To build:

```python
from lite50 import build_lite50_image_model
model = build_lite50_image_model(
    checkpoint_path="/path/to/original/efficientsam3_repvit.pt",
    device="cpu",
)
```

**Training is required**: New projection and FPN weights are not pretrained; MobileCLIP's low-rank embedding is approximate; layers have been pruned. First distill visual and text features from the original full RV-M teacher, then fine-tune on text grounding plus box/point prompted masks. Upstream Stage3 by default freezes FPN and its checkpoint may omit FPN weights: update optimizer, freeze logic and `skip_saving_parameters` before training. Optional `scripts/integration_stage3.py` gives a reviewable patch, not a complete trainer.

The text interface is `Sam3Processor.set_text_prompt`; box is `Sam3Processor.add_geometric_prompt([cx,cy,w,h], True, state)` (normalized), and point is `Sam3Processor.add_point_prompt([x,y], 1, state)` (pixel coordinates). **The separate optional SAM2 instance-interactive tracker is excluded from the 50M budget**. Do not claim open-vocabulary generalization until evaluated on held-out novel classes.

This overlay is original code inspired by the [official EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3) architecture. Refer to that project's Apache-2.0 license and the separate model-weight licenses.
