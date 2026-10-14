# Lite50：EfficientSAM3 进一步轻量化

当前项目入口：[efficientsam3-lite50](efficientsam3-lite50/README.md)。

数据准备看 [DATA_V2.md](efficientsam3-lite50/DATA_V2.md)，训练顺序看 [DISTILLATION.md](efficientsam3-lite50/DISTILLATION.md)。

主线：SA-1B 图像蒸馏 → Recap v3＋RefCOCO 系列文本蒸馏 → SA-1B 单点／单框几何蒸馏 → SACap 原文真实标注监督。

模型权重和大数据保存在用户 Drive，不提交到 GitHub。旧实验可从 Git 提交历史恢复，不作为当前训练入口。
