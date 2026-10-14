# Lite50 当前训练顺序

本页只描述当前主线。数据位置与验收条件见 [DATA_V2.md](DATA_V2.md)。

| 顺序 | 数据 | 更新模块 | 教师 |
|---|---|---|---|
| vision | SA-1B 图片 | 新 Projection、有效 FPN | 冻结 SAM3 |
| text | Recap v3＋RefCOCO 系列表达 | 压缩后的文本编码器 | 冻结 SAM3 |
| geometry | SA-1B 单正点／单框＋掩码 | Projection、有效 FPN | 冻结 SAM3 |
| stage3 | SACap 区域原文＋SA-1B 图像／掩码 | RepViT、Projection、有效 FPN | 不构建教师 |

保持当前默认损失和训练策略。各阶段载入前一阶段完整学生权重，重新创建对应优化器；同阶段恢复才使用 resume。不要把单步测试权重当作训练完成的权重。

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
