# EfficientSAM3 RV-M Lite50 — experimental code overlay

A **research prototype (not a trained checkpoint)** based on [EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3). Current goal: **CPU inference speed versus segmentation accuracy**, retaining text/open-vocabulary and single point/box interfaces; 50M parameters is no longer a hard constraint. Using the pinned upstream commit, the user instantiated 100,828,102 teacher parameters and **56,413,638 student parameters**. Mask accuracy has **not** been recovered or validated.

## Architecture changes

| Part | RV-M | Lite50 design |
|---|---|---|
| Backbone | RepViT-M1.1 | keep |
| Projection | 512→1024→dense 3×3 1024 | 512→256→DW3×3→1024 |
| FPN | Transposed convolution + four dense 3×3 convs | projection at low res, bilinear up/downsampling + depthwise-pointwise convs; only the first **three** used scales are executed |
| Text tokenizer | MobileCLIP-S0 vocabulary (49,408 entries) | unchanged |
| Text embedding | 49,408×512 | rank-128 factorization with optional SVD initialization (rank-96/64 ablations) |
| MobileCLIP-S0 attention | 4 transformer layers | 2, keeping first and last |
| Vision-language fusion encoder | 6 layers | 4 |
| Geometry encoder | 3 layers | 3 (unchanged; point/box path preserved) |
| Decoder, segmentation, scoring | original | unchanged |

The modified Projection, FPN, and rank-128 token embedding reduce the model to **56.41M** actual instantiated parameters (CPU validation; upstream commit `bd0936c788fed8d51fa799437f05abd97b401b06`). Text and fusion layers are also pruned, while all three Geometry Encoder layers and their pretrained weights remain. In the preliminary untrained 2-thread CPU test, Lite50 gave roughly **1.7×** end-to-end speedup compared with the RV-M teacher over a small set of prompts; **this does not establish accuracy or target-device speed**. The unused-FPN optimization below does **not** change the 56.41M parameter count.

## v0.2 decisions (2026-10-13)

- Default `Lite50Config.text_embedding_rank=128`; optionally test `96` or `64` for ablations.
- Default `Lite50Config.geometry_layers=3`; the complete original Geometry Encoder and its weights are kept. Reducing it to two layers is an **optional experiment only**, not the default.
- Input capabilities are **unchanged**: text concept prompts, point prompts, and normalized box prompts. No SAM2 multi-click instance tracker or video module has been added.
- Projection, FPN, text attention and vision-language encoder modifications are unchanged.
- No new pretrained weights, end-to-end tests, or quality metrics are provided by these config changes.
- Rank-128 adds **3,194,880 parameters** relative to rank-64. Current real size 56.41M is acceptable for the CPU speed/accuracy goal. The legacy 50M budget can be enforced explicitly using `strict_budget=True` or `scripts/report.py --strict-budget`.

## CPU inference optimization: skip the unused fourth FPN level

The pinned official model uses `Sam3DualViTDetNeck` with scales `[4, 2, 1, 0.5]`. Its `SAM3VLBackbone` has `scalp=1`: it **computes all four levels**, then discards the last (lowest-resolution) feature and its positional encoding.

The new `lite50/fpn_optimization.py` changes this ordering. `ActiveScaleFPN` computes **only levels 0, 1, 2**; the backbone's `scalp` is set to **0** so they are not truncated twice. The fourth branch remains registered but is never executed. This means:

- Three downstream FPN features, their shapes and expected values remain unchanged; all other input interfaces are untouched.
- **All v0.2 parameter names, counts and checkpoint keys are retained**: previous `student_initialized.pt` and later v0.2 `state_dict` files should load with `strict=True`.
- The change is enabled by default with `Lite50Config(skip_unused_fpn_level=True)`. Set `False` to run the unoptimized four-scale baseline and confirm parity.
- Standalone unit tests validate strict checkpoint compatibility, first-three-feature equality, unchanged parameter count and correct gradient connectivity. **End-to-end raw-output equality and real CPU speedup on the full upstream model still need to be checked in Colab.**
- The discarded scale only cost a small fraction of end-to-end CPU time in the previous profile. **Do not expect a large speed increase from this change alone.**

To compare full-model raw outputs (including the 200 unthresholded mask predictions) on the CPU, use the existing initialized v0.2 checkpoint:

```bash
python scripts/verify_fpn_skip.py \
  --student-weights /path/to/student_initialized.pt \
  --image /path/to/test_image.jpg \
  --modes text point box text_box
```

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
    device="cpu",  # speed/accuracy model (~56.41M); strict_budget=False by default
)
```

**Training is required**: New projection and FPN weights are not pretrained; MobileCLIP's low-rank embedding is approximate; text and fusion layers have been pruned (Geometry Encoder remains full-size). The fourth-FPN skip itself requires no retraining because the skipped output was already discarded. First distill visual and text features from the original full RV-M teacher, then fine-tune on text grounding plus box/point prompted masks. Upstream Stage3 by default freezes FPN and its checkpoint may omit FPN weights: update optimizer, freeze logic and `skip_saving_parameters` before training. Optional `scripts/integration_stage3.py` gives a reviewable patch, not a complete trainer.

The text interface is `Sam3Processor.set_text_prompt`; box is `Sam3Processor.add_geometric_prompt([cx,cy,w,h], True, state)` (normalized), and point is `Sam3Processor.add_point_prompt([x,y], 1, state)` (pixel coordinates). **The separate optional SAM2 instance-interactive tracker is excluded from the 50M budget**. Do not claim open-vocabulary generalization until evaluated on held-out novel classes.

This overlay is original code inspired by the [official EfficientSAM3](https://github.com/SimonZeng7108/efficientsam3) architecture. Refer to that project's Apache-2.0 license and the separate model-weight licenses.
