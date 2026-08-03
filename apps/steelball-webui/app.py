"""
钢珠检测交互页面 (Gradio Web UI) — 实时检测 + 帧率显示

功能：
  1. 上传钢珠图片 → 自动检测钢珠 + 直径测量
  2. 实时视频检测 → 摄像头/视频流逐帧检测 + 帧率显示
  3. 反光高光校验（可选）

状态：⚠️ 深化分支（需 gradio，钢珠模型 best.pt 就绪后运行）
依赖：pip install gradio (绕过代理: env -u http_proxy -u https_proxy -u all_proxy ...)

Usage:
    python apps/steelball-webui/app.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

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
                pd = ((x2 - x1) + (y2 - y1)) / 2  # 像素直径
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 0), 2)
                cv2.circle(annotated, (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                           3, (0, 165, 255), -1)  # 中心点
                cv2.putText(annotated, f"{conf:.2f} d={pd:.0f}px",
                            (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                info.append({"center": ((x1+x2)/2, (y1+y2)/2),
                             "diameter_px": pd, "conf": conf})
    return annotated, info


def video_stream(model, video_path):
    """实时视频检测：逐帧检测 + 帧率显示."""
    cap = cv2.VideoCapture(str(video_path))
    fps_target = cap.get(cv2.CAP_PROP_FPS) or 30

    frame_count = 0
    start = time.time()
    out_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, info = detect_steelballs(model, frame)
        frame_count += 1

        # 实时帧率
        elapsed = time.time() - start
        current_fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated, f"Balls: {len(info)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        out_frames.append(annotated)

        if frame_count >= 150:  # 最多处理150帧
            break

    cap.release()
    # 输出视频
    if out_frames:
        h, w = out_frames[0].shape[:2]
        out_path = Path("/tmp") / "steelball_detection.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_target, (w, h))
        for f in out_frames:
            writer.write(f)
        writer.release()
        return str(out_path), f"处理 {frame_count} 帧，平均 {frame_count/elapsed:.1f} FPS"
    return None, "无有效帧"


def webui():
    """Gradio 交互页面."""
    try:
        import gradio as gr
    except ImportError:
        print("需要安装 gradio: 绕过代理 pip install gradio")
        return

    model = load_model()

    def predict_image(image, real_diameter):
        """图片检测 + 直径测量."""
        if image is None:
            return None, "请上传图片"
        annotated, info = detect_steelballs(model, image)
        lines = [f"检测到 {len(info)} 个钢珠"]
        if real_diameter and info:
            k = real_diameter / info[0]["diameter_px"]
            lines.append(f"标定系数: {k:.5f} cm/px")
            for i, b in enumerate(info):
                lines.append(f"球{i}: 中心=({b['center'][0]:.0f},{b['center'][1]:.0f}), "
                             f"直径={b['diameter_px']:.0f}px ≈ {b['diameter_px']*k:.2f}cm, "
                             f"conf={b['conf']:.2f}")
        else:
            for i, b in enumerate(info):
                lines.append(f"球{i}: 中心=({b['center'][0]:.0f},{b['center'][1]:.0f}), "
                             f"像素直径={b['diameter_px']:.0f}px, conf={b['conf']:.2f}")
        return annotated, "\n".join(lines)

    def predict_video(video):
        """视频检测 + 帧率."""
        if video is None:
            return None, "请上传视频"
        out, summary = video_stream(model, video)
        return out, summary

    with gr.Blocks(title="钢珠检测交互页") as demo:
        gr.Markdown("# 钢珠检测 + 直径测量 + 实时检测\n"
                    "上传图片检测钢珠并测量直径；或上传视频看实时检测与帧率")
        with gr.Tab("图片检测"):
            with gr.Row():
                with gr.Column():
                    img_in = gr.Image(label="上传图片", type="numpy")
                    real_d = gr.Number(label="钢珠真实直径 (cm, 可选标定)")
                    btn_img = gr.Button("检测", variant="primary")
                with gr.Column():
                    img_out = gr.Image(label="检测结果")
                    txt_out = gr.Textbox(label="检测信息", lines=8)
            btn_img.click(predict_image, [img_in, real_d], [img_out, txt_out])

        with gr.Tab("视频检测"):
            with gr.Row():
                with gr.Column():
                    vid_in = gr.Video(label="上传视频", format="mp4")
                    btn_vid = gr.Button("开始检测", variant="primary")
                with gr.Column():
                    vid_out = gr.Video(label="检测结果视频")
                    txt_vid = gr.Textbox(label="处理信息")
            btn_vid.click(predict_video, [vid_in], [vid_out, txt_vid])

    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    webui()
