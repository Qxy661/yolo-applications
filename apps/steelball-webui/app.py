"""
钢珠检测交互页面 (Gradio Web UI) — 预留分支

功能：
  1. 上传钢珠图片 → 自动检测钢珠
  2. 显示检测结果（bbox + 置信度）
  3. 直径测量（可选：提供真实直径做标定）

状态：⚠️ 预留分支框架（需 gradio，钢珠模型 best.pt 就绪后运行）
依赖：pip install gradio (绕过代理: env -u http_proxy -u https_proxy -u all_proxy ...)

Usage:
    python apps/steelball-webui/app.py
"""
import sys
from pathlib import Path

import cv2

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "runs" / "steelball" / "steelball_yolo26" / "weights" / "best.pt"
MODEL_PATH = ROOT / "runs" / "steelball" / "best.pt"  # fallback 路径

from ultralytics import YOLO


def load_model():
    """加载钢珠检测模型（best.pt 优先）."""
    for p in [
        ROOT / "runs" / "steelball" / "steelball_yolo26" / "weights" / "best.pt",
        ROOT / "runs" / "steelball" / "weights" / "best.pt",
        ROOT / "weights" / "best.pt",
    ]:
        if p.exists():
            print(f"加载模型: {p}")
            return YOLO(str(p))
    raise FileNotFoundError("未找到钢珠模型 best.pt")


def detect_steelballs(model, image_bgr):
    """检测钢珠，返回 (结果图, 检测信息)."""
    results = model(image_bgr)
    annotated = image_bgr.copy()
    info = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                # 像素直径 ≈ bbox 平均边
                pd = ((x2 - x1) + (y2 - y1)) / 2
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.putText(annotated, f"{conf:.2f} d={pd:.0f}px",
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                info.append(f"钢珠: 中心=({(x1+x2)/2:.0f},{(y1+y2)/2:.0f}), "
                            f"像素直径={pd:.0f}px, 置信度={conf:.2f}")
    return annotated, info


def webui():
    """Gradio 交互页面."""
    try:
        import gradio as gr
    except ImportError:
        print("需要安装 gradio: 绕过代理 pip install gradio")
        return

    model = load_model()

    def predict(image, real_diameter):
        if image is None:
            return None, "请上传图片"
        annotated, info = detect_steelballs(model, image)
        # 标定（可选）
        lines = [f"检测到 {len(info)} 个钢珠"]
        if real_diameter and info:
            # 用第一个球标定
            import re
            m = re.search(r"像素直径=(\d+)", info[0])
            if m:
                pd = int(m.group(1))
                k = real_diameter / pd
                lines.append(f"标定系数: {k:.5f} cm/px")
                lines += [x.replace("px", f"px ({float(real_diameter):.2f}cm)" if "像素" in x else "")
                          for x in info]
        lines += info
        return annotated, "\n".join(lines)

    with gr.Blocks(title="钢珠检测交互页") as demo:
        gr.Markdown("# 钢珠检测 + 直径测量\n上传钢珠图片，自动检测并测量直径")
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(label="上传图片", type="numpy")
                real_d = gr.Number(label="钢珠真实直径 (cm, 可选标定)")
                btn = gr.Button("检测", variant="primary")
            with gr.Column():
                img_out = gr.Image(label="检测结果")
                txt_out = gr.Textbox(label="检测信息", lines=8)
        btn.click(predict, [img_in, real_d], [img_out, txt_out])

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    webui()
