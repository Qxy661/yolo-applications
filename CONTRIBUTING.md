# Contributing Guide

感谢你考虑为 YOLO 小目标检测项目做贡献！本指南帮助你了解项目结构和贡献流程。

## 项目结构

```
yolo-visdrone/
├── src/                 # 核心代码（训练/评估/SAHI/优化）
├── scripts/             # 应用脚本（钢珠训练/部署等）
├── data/                # 数据集配置（yaml）
├── docs/                # 文档
│   ├── tutorial/        # 知识体系（学习态）
│   └── ...              # 工程文档（展示态）
├── applications/        # 具体应用（steel-ball / wind-turbine）
├── results/             # 实验指标
└── runs/                # 训练输出（gitignored）
```

## 环境

```bash
pip install -r requirements.txt
```

## 如何贡献

1. **报告问题**：提 Issue，描述清楚复现步骤
2. **提交代码**：
   - Fork 仓库
   - 创建分支：`feature/xxx`
   - 提交改动（遵循 commit 规范）
   - 发起 Pull Request

## Commit 规范

- 动词开头：`Add` / `Fix` / `Refactor` / `Update`
- 简洁描述改动：`Add steel-ball training script`

## 代码规范

- Python 遵循 PEP 8
- 关键逻辑写清楚注释
- 新功能带可复现的命令/说明

## 文档规范

- 教学文档（tutorial/）：中文 + 英文代码注释
- 工程文档：简洁精炼
- 每篇文档写清前置和下一步

## 测试

- 训练/评估前确认数据路径正确
- 提交前跑通关键脚本

## License

本项目使用 MIT License（见 [LICENSE](LICENSE)）。
