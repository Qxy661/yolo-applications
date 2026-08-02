# 钢珠检测交互页面（预留分支）

> 基于 Gradio 的钢珠检测 Web UI——部署实操的展示应用。
> **状态：预留分支**，框架已搭好，待钢珠模型就绪 + gradio 安装后运行。

## 功能

1. **上传图片** → 自动检测钢珠
2. **显示结果**：检测框 + 置信度 + 像素直径
3. **直径测量**：可选提供真实直径做标定（cm/px 系数）

## 运行

```bash
# 1. 安装 gradio（绕过代理）
env -u http_proxy -u https_proxy -u all_proxy \
  -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  pip install gradio -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 确认钢珠模型 best.pt 存在（runs/steelball/）
# 3. 启动
python apps/steelball-webui/app.py
# 浏览器访问 http://localhost:7860
```

## 技术点

- **Gradio**：快速构建 Web UI（上传/展示/交互）
- **YOLO 推理**：加载 best.pt 实时检测
- **直径测量**：bbox → 像素直径 → 标定 → 物理直径
- **反光校验**（可选扩展）：可接入高光点校验（trick）

## 为什么用钢珠做部署

- 钢珠是**单类简单应用**，部署最直接（体现闭环完整）
- 直径测量是**检测 + 测量的应用延伸**（不只是框出目标）
- 交互页面让"部署"可演示（上传图片即检测），作品集效果好

## 与主流程关系

- 这是**应用流程闭环的最后一步（部署）**的展示
- 钢珠分支的部署延伸（检测 → 测量 → 交互）
- 交互页面是**可选增强**，核心还是模型 + 推理
