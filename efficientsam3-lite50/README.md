# EfficientSAM3 Lite50

面向 CPU 推理的研究模型。学生约 **56.41M** 参数；教师为完整 SAM3，初始化来自官方 RV-M。尚未完成正式蒸馏和精度验收，不是已经训练好的模型。

## 从哪里开始

| 任务 | 入口 |
|---|---|
| 下载、安装和核验训练数据 | [DATA_V2.md](DATA_V2.md) |
| 四阶段训练、保存和恢复权重 | [DISTILLATION.md](DISTILLATION.md) |
| 模型定义 | `lite50/builder.py`、`lite50/modules.py` |
| 首批 500 张训练及恢复 | `scripts/run_mini500_pilot.py`（已修复分块 BN，非全量配方） |
| 数据准备及训练脚本 | `scripts/` |
| 回归测试 | `tests/` |
| 训练配置 | `configs/distill_geometry.yaml`、`configs/distill_stage3.yaml` |

## 当前训练方案

1. **视觉蒸馏**：SA-1B 图片，SAM3 教师特征。
2. **文本蒸馏**：Recap v3 短语＋RefCOCO／RefCOCO+／RefCOCOg 原始表达，教师和学生接收相同文本，使用官方 16-Token Tokenizer。
3. **几何蒸馏**：SA-1B 单正点或单框、真实掩码与教师监督。
4. **Stage 3**：SACap 原始区域文本＋SA-1B 原图和真实掩码；只在 Tokenizer 内按 16 Token 处理，不用旧短语白名单，不运行教师前向。

Stage 3 保持现有官方式配置：训练 RepViT、Projection 和有效 FPN；冻结文本、Fusion、Geometry、解码器和分割头。首批测试已修正提示分块的 BN 重复更新；当前证据支持优先排查视觉特征适配，暂不扩大默认 Fusion 解冻范围。详见 [首批测试后的修正](DISTILLATION.md#首批-500-张测试后的修正)。

## 学生结构

保留 RepViT-M1.1、完整三层 Geometry Encoder、Decoder、Segmentation 和 Scoring；修改 Projection 和 FPN，文本 Embedding 使用 rank-128 分解，文本 Transformer 保留两层，Fusion 保留四层。FPN 注册四个分支以兼容旧权重，只执行前三个有效尺度。

保留文本、单点和单框接口，不增加 SAM2 多轮掩码修正模块。接口可运行不等于精度保留，最终需要独立文本／点框精度与目标 CPU 延迟测试。

## 环境与来源

官方仓库：[SimonZeng7108/efficientsam3](https://github.com/SimonZeng7108/efficientsam3)，固定提交 `bd0936c788fed8d51fa799437f05abd97b401b06`。安装官方依赖，再安装本目录的 `requirements-distillation.txt`。

将本目录加入 `PYTHONPATH`，并加入官方仓库及其 `sam3/` 子目录。执行：

```bash
python -m unittest discover -s tests -v
python scripts/report.py --checkpoint /path/to/efficientsam3_repvit.pt
```

真实 SAM3 集成测试需要安装完整依赖并设置 `SAM3_ROOT`。未执行的测试不得算作通过。

模型和数据保留各自上游许可。旧 Fusion/KD 实验配置、SACap 短语白名单读取器及废弃补丁入口已从当前文件树移除；旧版本保留在 Git 历史中。教师缓存、训练审计、模型导出和当前测试继续保留。
