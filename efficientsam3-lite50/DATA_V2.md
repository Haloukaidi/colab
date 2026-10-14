# 当前训练数据入口：Recap v3 + RefCOCO；Stage 3 使用 SACap 原文

本说明对应 2026-10-14 的用户数据方案。只适配数据，不修改 Lite50 结构、16-Token Tokenizer、损失或默认 Stage 3 可训练模块。

## 各阶段用什么

- 图像特征蒸馏：SA-1B 图像，冻结的完整 SAM3 为教师。
- 文本特征蒸馏：已经整理的 Recap v3 独立短语 + RefCOCO / RefCOCO+ / RefCOCOg 原始表达。同一字符串交给教师和学生，两者均使用官方 ctx16。不要再把长 caption 的教师缓存与短语学生输入混用。
- 几何提示蒸馏：沿用 SA-1B 的单正点、单框和真实掩码。
- Stage 3：保留官方式纯监督路线，使用 SACap 原始 `segments_info[].caption` + 对应 SA-1B 原图和原始区域掩码。无教师前向，文本提示、不输入提示框。训练 RepViT/Projection/有效 FPN，文本/Fusion/Geometry/解码及分割模块冻结。

SACap 不再预先提取短语，不使用 `short_text`，不要求 `enabled_for_training` 人工审核白名单。超长文字由官方 `SimpleTokenizer` 在前向时截断；原文和原始掩码不编辑。结构配对校验仍必须通过。历史短语白名单数据只保留作审计，不作为默认 Stage 3 入口。

## 1. 文本包

`TextManifestDataset` 读取 Parquet 或 JSONL，必须有 `text/split/source`。

已生成的混合文本：训练 1,225,708 条、验证 21,166 条、测试 24,916 条。训练来源为 Recap v3 1,038,999 条，以及 RefCOCO 系列 186,709 条。首批开发子集为 9,000 条训练 + 1,000 条验证。

原始 RefCOCO 三系列共下载 378,784 条表达，使用 UNC/UNC/UMD 划分。由于原站下载链接失效，来源是固定版本研究镜像，源文件 SHA256 与完整源记录已保留。这不是作者未公开训练清单的精确复刻；本次未加入 RF100-VL，纯文本阶段不需要重新下载全部 COCO 图片。

混合前隔离留出图像/原始划分，混合时排除规范化文本及实际 ctx16 输入重复。完整训练按保留的唯一有效输入逐条访问，`frequency` 仅作来源统计，不自动重复采样。首批文本为了调试两类来源，采用 Recap/RefCOCO 各半的开发采样；这不是论文声称的官方比例。

复建使用 `scripts/prepare_mixed_text_v2.py`，输入为冻结的 Recap v3 `train_unique_phrases.parquet`、`nontraining_phrase_frequencies.json` 和原始 RefCOCO 文本包。需要固定 Tokenizer 源码与 BPE 词表。

## 2. 首批 500 张

复用已经保存的 `mini500_基础图像与原始标注_v1.zip`，不重复生成或下载 563 MB 图像包。另一个 v2 补充包提供新文本清单、原始样本清单和审查记录。

```bash
# 先解压 v2 补充包到 $PATCH；$CODE 为本目录。
python "$CODE/scripts/install_mini500_v2.py" \
  --base-zip "$BASE_ZIP" --patch-dir "$PATCH" \
  --output /content/lite50_distill_mini_v2
```

安装器校验基础 ZIP SHA256、500 张图像、4,049 条点框掩码与 409 条 SACap 区域原文；保留基础原始文件，只新增 `manifests/text_mixed_v2.parquet`。出现任何不一致即停止，不覆盖不同的现存数据。完成后应看到 `DATA_V2_READY.json`。

图像/几何阶段只改 `dataset_root`。文本阶段明确传入新清单：

```bash
python "$CODE/scripts/train_feature_distillation.py" \
  --official-root "$OFFICIAL" --dataset-root /content/lite50_distill_mini_v2 \
  --phase text --text-manifest /content/lite50_distill_mini_v2/manifests/text_mixed_v2.parquet \
  --teacher-checkpoint "$SAM3" --rv-checkpoint "$RVM" \
  --lite-checkpoint "$VISION_CKPT" --output /content/lite50_training/text
```

Stage 3 沿用纯监督配置，只修改数据和前一阶段权重路径：

```bash
python "$CODE/scripts/train_distill_stage3.py" --official-root "$OFFICIAL" \
  paths.dataset_root=/content/lite50_distill_mini_v2 \
  paths.lite_checkpoint="$GEOMETRY_CKPT" \
  paths.output=/content/lite50_training/stage3 epochs=1
```

上述 1 轮为开发检查，不是正式训练充分性的声明。首批只有 64 张图像有 SACap 配对，训练 58 张/375 区域、验证 6 张/34 区域；不能用它证明最终泛化精度。

## 3. 官方 1% 清单与自选 5% 归档

1% 归档严格是 `sa_000020.tar` 到 `sa_000029.tar`，原始备份来源与前次云端哈希验证保留在清单。匹配 SACap 原始 10,388 张图像 / 59,408 区域。

自选 5% 是用户固定 50 个完整归档，并非作者未公开的精确 SA-1B-5P。匹配 SACap 原始 51,355 张图像 / 292,987 区域。

两个文字范围已重新从固定版本的公开 SACap 源表生成，源表完整 SHA256 已核验，所有区域原文逐条保持不变。已标记已知 SA-Co 测试图像重叠：1% 排除 1 图/7 区域，5% 排除 6 图/32 区域。保留首批已存在的图像划分，其余图像按公开的 seed123 哈希规则作约 99/1 划分，不能冒称是作者原始划分。

原始 TAR 就是完整图像/JSON 数据，不为归档组织而复制数百 GB 文件。SACap 配套包包含 `sacap/{train,val}.parquet`、`manifests/sacap_regions.parquet`、范围清单和源校验。

文字清单成功不等于全量图像/掩码已重新验证。读取本地原始 TAR、提取 Stage 3 对应媒体并完整核验：

```bash
python "$CODE/scripts/materialize_sacap_scope.py" \
  --scope-dir "$SCOPE_PACKAGE" --archives-root "$EXISTING_TAR_FOLDER" \
  --output /content/lite50_stage3_scope
```

只有 `materialization_audit.json` 为 `passed` 且 `training_ready=true` 才能把该输出作为正式 Stage 3 数据根目录。该脚本提取 SACap 匹配媒体，不代表解包整个 SA-1B；原始归档仍保留完整视觉/几何数据。可选 `--verify-archive-hash` 会读取整个归档校验有记录的哈希。

官方 1% SACap 交集此前已有 Colab 全图像/RLE 校验记录；自选 5% 此次仍未重新读取全部 50 个大归档，不能声称全量掩码通过。正式 5% 训练前须完成上述本地核验。

## 审查范围

数据处理、来源哈希、分割隔离、原文不改写、Dataset 输入和单元测试都有报告。没有在本次任务中蒸馏模型，也没有证明精度提升或实际端到端加速。Recap v3 的独立短语仍可能丢失关系/否定等语义；它适合纯文本特征蒸馏，不是图像目标存在性标注。
