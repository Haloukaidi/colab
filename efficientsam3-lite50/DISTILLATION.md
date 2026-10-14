# Lite50 staged distillation

Full **SAM3 (`sam3.pt`) is the frozen teacher**. RV-M
`efficientsam3_repvit.pt` only initializes the student. This overlay targets
official commit `bd0936c788fed8d51fa799437f05abd97b401b06`, without modifying
the official checkout. Smoke tests establish update connectivity, not accuracy.

## Curriculum

| Phase | Trainable modules |
|---|---|
| `vision` | New Projection and active FPN 0/1/2 |
| `text` | Retained MobileCLIP-S0 text branch, including rank-128 codes and projection |
| `geometry` | New Projection and active FPN 0/1/2; text and all other modules frozen |
| `fusion1` | Projection, FPN 0/1/2, text, last Fusion layer |
| `fusion2` | Same, with last 2 Fusion layers |
| `fusion4` | Same, with all 4 Fusion layers |

RepViT, Geometry Encoder, Transformer Decoder, scoring and segmentation heads,
and unused FPN 3 remain frozen. Frozen modules stay in evaluation mode, including
BatchNorm running statistics. Their forward passes remain differentiable with
respect to trainable upstream features; only teacher computation uses `no_grad`.

The `geometry` phase comes **after text, before fusion1**, matching the official Stage1 Geometry Fine-tuning placement. It trains **one positive point OR one box** on SA-1B ground-truth regions (not repeated clicks and not SAM2 interaction). The official collator's point data are normalized `(x,y,label)`; the Lite50 adapter converts them to real `Prompt.point_embeddings`/`point_labels`/`point_mask` and calls `forward_grounding` directly. Official `SAM3Image.forward` otherwise warns that PCS ignores points. Box and point samples can coexist in the same batch, with an explicit 3-D point padding width rather than the PCS default of 257. The teacher and student receive exactly the same geometry. Supervised GT mask losses, image feature KD, and filtered teacher mask KD remain active. Geometry-only input uses the official `visual` text placeholder.

Each phase loads the previous full student checkpoint and builds a fresh
optimizer. Parameter IDs must exactly cover `requires_grad=True`, without frozen
or duplicate parameters. Stage3 uses official `construct_optimizer` with an
explicit allowlist. Default LRs: Projection/FPN 1e-4, text/Fusion 1e-5. These
are starting settings, not validated convergence hyperparameters. No ViT
layer-decay rule is applied to RepViT wrappers. Zero-LR groups are rejected.

## Reused objectives and framework

* Stage1 masked image MSE/cosine and padding-aware text MSE/cosine formulas are
  preserved in `lite50/distillation.py`, avoiding import-time DDP/global module
  dependencies of the original entrypoints. Image MSE sums channels, while text
  MSE averages channels. Image MSE is weighted 0.0015 and cosine 0.1.
* New FPN outputs also receive mean MSE/cosine supervision, weighted 0.1;
  trunk distillation alone cannot update FPN. Distilled locations are the
  post-Projection/teacher ViT trunk `[B,1024,72,72]`, active FPN scales
  `[B,256,288/144/72,288/144/72]`, and resized text memory `[16,B,256]`.
* Both models receive the same 1008-square image and SAM3 normalization
  `(image/255 - .5)/.5`. Square warping has no padding; the image valid mask is
  all ones. Letterboxing would require changing that mask. Resolution is fixed
  because the RV-M wrapper interpolates projected features to 72x72.
* Both text encoders use the official BPE and 16-token context (14 content
  tokens plus start/end). The full teacher keeps its pretrained position table
  and shortens its input context. Padding masks must match. Recap uses identical
  built-in truncation for teacher/student; SACap phrases over budget are errors.
* Geometry mask KD directly imports upstream `GeometryFinetuningLoss` for
  BCE + Dice on teacher logits. Predictions are independently Hungarian-matched
  to the same GT region, never by query index. Only teacher masks with GT IoU
  >= 0.3 become soft targets. `teacher_mask_pairs=0` means KD was gated off.
* Stage3 subclasses the actual `sam3.train.trainer.Trainer`, preserving its DDP,
  epoch loop, AMP option, clipping, optimizer schedulers and checkpoints.
  Official Box/GIoU, classification/presence and focal/Dice losses remain active;
  feature KD and gated Geometry mask KD are added to `core_loss`.
* The standalone YAML replaces the absent upstream `base_train.yaml` dependency
  and incompatible model/data arguments in the pinned Stage3 configurations.
  The old partial `integration_stage3.py` patch is retired.

## SA-1B geometry interface

`SA1BGeometryStage3Dataset` reads all validated `manifests/geometry.jsonl` rows from mini-v1 (3,647 train, 402 val). Its `point` mode sends exactly one positive foreground point; `box` sends one mask-derived bounding box; `mixed` alternates by sample index and epoch. Official mask decoder, synchronized 1008-square image/mask/point/box transforms, and SAM3 Stage3 collator are reused. These inputs train the point and box *grounding* path only; they do not implement iterative SAM2 mask correction. Positive+negative or multiple points are supported by the lower-level Prompt representation but are not yet covered by this one-point curriculum and require later training/validation.

Use `configs/distill_geometry.yaml` with the existing `scripts/train_distill_stage3.py`, passing the text-phase checkpoint as `paths.lite_checkpoint`. Keep `accelerator=cpu workers=0 limit=2 epochs=1 trainer.audit_updates=true` only for a **two-sample point+box smoke test**, then train normally. Do not promote a smoke checkpoint to a trained model.

## SACap interface

`SACapStage3Dataset` supports native `sacap/{train,val}.parquet` with `imagename`,
`caption`, and `segments_info=[{anno_id, caption}]`. It joins image stem +
`anno_id` to the final short-prompt sidecar and original SA-1B JSON. Checks cover
source captions, fixed split, image path, region ID, exact RLE, positive mask
area, boxes, image dimensions and token budget. Missing masks are errors; boxes
are never substituted for masks.

Default eligibility requires `accepted`/`corrected` plus `enabled_for_training`:
31 training and 14 validation regions in the current sidecar. These were visually
reviewed by the assistant, not independently human-annotated. Unreviewed automatic
candidates need explicit `include_unreviewed=true`; held/rejected pairs stay
excluded. The existing train/val split is preserved.

Each sample is one reviewed region. `prompt_mode=mixed` rotates text, box
(`visual` placeholder), and text+box by index/epoch. Official transforms update
image, mask and box together before the official collator. SACap annotations are
non-exhaustive, so other instances are not automatically negative labels.
The SACap fusion-stage adapter still uses `text`, `box`, `text_box`. **Point prompting is trained in the preceding SA-1B geometry phase**, because the official PCS training forward ignores points. Inference point support and frozen Geometry weights remain.

## Colab commands

Use the overlay alongside the pinned official checkout. Install official model
and training requirements plus `pyarrow`, `pycocotools`, `hydra-core`, `omegaconf`,
`fvcore`, `tensorboard`, `torchmetrics`, `triton`, `psutil`. The pinned official
loss module imports Triton even on CPU; its CPU loss computation uses PyTorch.
spaCy is not required during training.

```bash
OFFICIAL=/content/lite50_v02_validation/official
OVERLAY=/content/lite50_v02_validation/lite50_source/efficientsam3-lite50
DATA=/content/lite50_distill_mini_v1
WEIGHTS='/content/drive/MyDrive/EfficientSAM3压缩/原始权重'
cd "$OVERLAY"

python scripts/train_feature_distillation.py --official-root "$OFFICIAL" \
  --dataset-root "$DATA" --teacher-checkpoint "$WEIGHTS/sam3.pt" \
  --rv-checkpoint "$WEIGHTS/efficientsam3_repvit.pt" --phase vision \
  --steps 1000 --output /content/lite50_distillation/vision

python scripts/train_feature_distillation.py --official-root "$OFFICIAL" \
  --dataset-root "$DATA" --teacher-checkpoint "$WEIGHTS/sam3.pt" \
  --rv-checkpoint "$WEIGHTS/efficientsam3_repvit.pt" --phase text --lr 1e-5 \
  --lite-checkpoint /content/lite50_distillation/vision/checkpoint.pt \
  --steps 1000 --output /content/lite50_distillation/text

# SA-1B geometry KD: single positive point and single box, alternating.
python scripts/train_distill_stage3.py --official-root "$OFFICIAL" \
  --config configs/distill_geometry.yaml \
  paths.lite_checkpoint=/content/lite50_distillation/text/checkpoint.pt \
  paths.output=/content/lite50_distillation/geometry epochs=1

# Then switch to reviewed SACap text/box fusion fine-tuning.
python scripts/train_distill_stage3.py --official-root "$OFFICIAL" \
  phase=fusion1 paths.lite_checkpoint=/content/lite50_distillation/geometry/checkpoints/checkpoint.pt \
  paths.output=/content/lite50_distillation/fusion1 epochs=1

python scripts/train_distill_stage3.py --official-root "$OFFICIAL" \
  phase=fusion2 paths.lite_checkpoint=/content/lite50_distillation/fusion1/checkpoints/checkpoint.pt \
  paths.output=/content/lite50_distillation/fusion2 epochs=1

python scripts/train_distill_stage3.py --official-root "$OFFICIAL" \
  phase=fusion4 paths.lite_checkpoint=/content/lite50_distillation/fusion2/checkpoints/checkpoint.pt \
  paths.output=/content/lite50_distillation/fusion4 epochs=1
```

Step counts/epochs above are examples, not evidence of sufficiency. Advance phases
after checking held-out masks and forgetting. The provided Stage3 configuration
is `train_only`; it does not replace a segmentation accuracy benchmark.

For CPU smoke tests use `--device cpu` for feature training or
`accelerator=cpu workers=0 limit=1 trainer.audit_updates=true` for Stage3.
Full 1008-resolution FP32 teacher/student training requires substantial RAM.
Frozen teacher weights use strict `assign=True` loading from the memory-mapped
checkpoint. On CPU this keeps their storage file-backed and reclaimable during
student backward; freed constructor allocations are also returned to the OS when
glibc supports `malloc_trim`. CUDA materializes normal device weights. The checkpoint must
remain available and immutable while that CPU teacher is alive.
CUDA is the default for real training; AMP is explicit opt-in after validation.
CPU runs substitute process RSS for the upstream CUDA-only memory meter, through
a scoped runtime adapter; they do not report RSS as GPU memory.
They also select the official focal loss's existing PyTorch branch instead of
its default GPU-only Triton kernel. CUDA runs retain the upstream kernel.

Optional `trainer.model.teacher_cache_dir=/path/to/cache` caches frozen teacher
features and raw predictions. Keys include a hash of every teacher weight/buffer,
official version, context, exact normalized image tensor, text, query mapping,
boxes, labels and padding masks. GT matching is recomputed after every cache load.
Augmented inputs produce different keys. Cache files can be large; caching is
disabled by default. Do not mutate teacher weights within a cache instance.

## Checkpoints and resume

Feature checkpoints contain the complete student, optimizer, phase, global step,
seed and Torch RNG state. `--resume` continues the same phase/dataloader position;
`--lite-checkpoint` starts a new phase with a fresh optimizer.
Stage3 retains complete upstream checkpoints (`skip_saving_parameters: []`),
including student, optimizer, epoch, losses and optional scaler, plus Lite50
phase/teacher/context/commit provenance. Resume with a different phase is rejected.
Use a new output directory per phase. The teacher is excluded from student
checkpoints and must remain available.

Stage3 model keys start with `student.`. `lite50.stage3.build_student` accepts
both Stage3 and feature checkpoints, strips this prefix and loads strictly.
For the inference-only builder, export this model's plain `state_dict` first.
Stage3 resume keeps upstream epoch-boundary semantics, without promising exact
replay of arbitrary mid-epoch augmentations/dropout. Smoke checkpoints have only
a few updates and are not trained production models.

## Validation

`scripts/validate_feature_distillation.py` checks real teacher/student weights,
native SACap train/val data and three prompt modes, one image and text update,
finite/nonzero gradients, active-group changes, frozen state fingerprints, and
exact feature-checkpoint reload. `trainer.audit_updates=true` adds corresponding
audits to an actual Stage3 epoch, including full teacher state fingerprints.
Use `limit=1` for this expensive integration check.

Unit tests cover phase policies, missing optimizer coverage, padding gradients,
independent query matching and gating of inconsistent teacher masks. Executed
results are recorded separately; code alone is not evidence of passing tests.
