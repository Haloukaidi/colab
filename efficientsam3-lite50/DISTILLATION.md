# Lite50 当前训练顺序

本页只描述当前主线。数据位置与验收条件见 [DATA_V2.md](DATA_V2.md)。

| 顺序 | 数据 | 更新模块 | 教师 |
|---|---|---|---|
| vision | SA-1B 图片 | 新 Projection、有效 FPN | 冻结 SAM3 |
| text | Recap v3＋RefCOCO 系列表达 | 压缩后的文本编码器 | 冻结 SAM3 |
| geometry | SA-1B 单正点／单框＋掩码 | Projection、有效 FPN | 冻结 SAM3 |
| stage3 | SACap 区域原文＋SA-1B 图像／掩码 | RepViT、Projection、有效 FPN | 不构建教师 |

当前默认损失和冻结策略是待验证基线，不代表通过精度验收。各阶段载入前一阶段完整学生权重，重新创建对应优化器；同阶段恢复才使用 resume。不要把单步测试权重当作训练完成的权重。

## 开始前

安装官方固定提交 `bd0936c788fed8d51fa799437f05abd97b401b06` 及依赖。以下命令在本项目目录运行。准备环境变量：`OFFICIAL` 为官方仓库路径、`DATA` 为通过核验的数据根目录、`SAM3` 为 `sam3.pt`、`RVM` 为 `efficientsam3_repvit.pt`、`OUT` 为新的训练输出目录。

首批安装必须通过 `install_mini500_v2.py`，生成 `DATA_V2_READY.json`。5% 的文字标注包不是完整媒体包；`materialize_sacap_scope.py` 必须完成核验后再训练。

## 1. 图像特征蒸馏

```bash
python scripts/train_feature_distillation.py \
  --official-root "$OFFICIAL" --dataset-root "$DATA" \
  --teacher-checkpoint "$SAM3" --rv-checkpoint "$RVM" \
  --phase vision --output "$OUT/vision"
```

## 2. 文本特征蒸馏

```bash
python scripts/train_feature_distillation.py \
  --official-root "$OFFICIAL" --dataset-root "$DATA" \
  --teacher-checkpoint "$SAM3" --rv-checkpoint "$RVM" \
  --phase text --text-manifest "$DATA/manifests/text_mixed_v2.parquet" \
  --lite-checkpoint "$OUT/vision/checkpoint.pt" --output "$OUT/text"
```

全量文本也可通过 `--text-manifest` 指定；必须使用相应训练划分。教师和学生看到同一份文本，不复用原长描述生成的教师缓存来监督不同短语。

## 3. 单点／单框几何蒸馏

```bash
python scripts/validate_geometry_inputs.py --official-root "$OFFICIAL" --dataset-root "$DATA"
python scripts/train_distill_stage3.py --official-root "$OFFICIAL" \
  --config configs/distill_geometry.yaml \
  paths.dataset_root="$DATA" paths.rv_checkpoint="$RVM" paths.teacher_checkpoint="$SAM3" \
  paths.lite_checkpoint="$OUT/text/checkpoint.pt" paths.output="$OUT/geometry"
```

几何阶段使用单正点或单框，不做多轮点击。点必须真实进入 Geometry Encoder，不能经忽略点输入的普通 PCS 前向。这个适配由现有代码实现。

## 4. SACap 原文监督微调

```bash
python scripts/train_distill_stage3.py --official-root "$OFFICIAL" \
  paths.dataset_root="$DATA" paths.rv_checkpoint="$RVM" \
  paths.lite_checkpoint="$OUT/geometry/checkpoints/checkpoint.pt" \
  paths.output="$OUT/stage3"
```

使用 `RawSACapStage3Dataset`、文本提示和原始区域 caption；超长文字在官方 Tokenizer 内截断，不离线改写。损失使用现有官方框、分类／存在性和 Mask／Dice 监督，不构建 SAM3 教师。

当前配置的 1 轮是开发默认，不代表正式训练已充分；官方 RV-M 参考实验为 20 轮。50 个归档是用户固定样本，不是作者未公开精确 5% 清单。

## 必要检查

每阶段检查有限损失、有效梯度、正确优化器覆盖、冻结参数保持不变，以及 checkpoint 严格重新加载。单步更新通过不代表精度恢复。

`validate_feature_distillation.py` 使用当前混合文本和 SACap 原文作数据／更新检查；`validate_geometry_inputs.py` 检查点框数据；`verify_distillation_checkpoints.py` 检查权重。最终在独立数据上测文本分割、单点和单框精度，并在同一目标 CPU 上比较延迟、峰值内存。

训练框架复用官方 Stage 1 特征蒸馏及 Stage 3 Trainer／监督损失；Lite50 构建与数据接口是适配层，不等于作者未公开训练数据与权重的逐字节复现。

## 首批 500 张测试后的修正

已确认的修复是提示分块的 BN 统计更新：同一图像的多个提示分块只保留一次 running_mean、running_var 和 num_batches_tracked 更新。训练态归一化与 affine 参数梯度仍启用。`lite50.bn_once.BatchNormOncePerImage` 必须在每个分块 backward 后调用，每张新图像重新创建；不要在 forward 与 backward 之间修改 BN buffer，也不要跨图像复用实例。各分块必须使用同一张预处理后的图像。

`scripts/run_mini500_pilot.py` 已接入修复。它是已准备好的首批数据专用入口，默认工作目录 `/content/lite50_experiment`，包含 `mini500/`、`official/`、`sam3.pt` 和 `efficientsam3_repvit.pt`；源码从当前脚本所在仓库加载。先按前文安装依赖和首批数据。使用已有实验输出目录可恢复模型、优化器、调度位置及随机状态：

```bash
python scripts/run_mini500_pilot.py --workdir /content/lite50_experiment --output /path/to/existing-mini500-output
```

已完成的阶段会跳过，不会因再次执行而重新蒸馏。仅在确实要运行首批测试时执行此命令。此入口有首批训练记录数限制，不用于全量训练。其中 50 步预热和固定调度总步数属于实验设置，尚未按全量清单验证；正式训练前必须重新推导，不能直接沿用为完整配方。普通 Stage 3 Trainer 入口没有新增提示微分块，仍沿用原训练流程。

2026-10-14 小规模验证：

- 两张真实图、各 8 个提示：4+4 分块修复后，BN buffers 和更新权重前的评估输出与整批前向一致，梯度余弦大于 0.999；不宣称 BF16 梯度逐位相同。
- 相同 24 张验证图、每图一个区域，最终学生单点/单框 IoU 为 6.94%/4.51%；仅替换原版视觉模块为 25.91%/60.93%，仅替换教师视觉模块为 81.39%/90.32%，仅恢复原版融合模块为 7.57%/7.83%。这些是诊断替换，不是压缩学生实际达到的精度。交换视觉模块必须同步 `backbone.scalp`，保证三个有效 FPN 尺度一致。
- 从最终 Stage 3 第 290 步接续，两组各训练 32 步、使用相同 16 张训练图和修复后的 BN。另外 26 张验证图、208 个区域上，原模型单点/单框 IoU 为 3.81%/4.19%，融合冻结组为 4.93%/4.01%，仅解冻融合最后一层为 4.82%/4.10%。解冻没有形成稳定几何精度优势，因此不修改默认冻结策略。
- IoU 按置信度选 top-1 掩码，未作置信度过滤；不同验证子集不能混合比较。测试为短程、单随机种子，不代表全量效果。梯度、冻结状态、优化器覆盖和保存重载检查通过，原检查点保留。

当前优先排查视觉投影/FPN 的特征方向和幅值，并配合独立掩码 IoU；不因 MSE 或训练损失下降就判定压缩完成。视觉替换、融合解冻和事后 BN 校准均不作为新的默认结构或正式权重。本次代码更新不触发训练。
