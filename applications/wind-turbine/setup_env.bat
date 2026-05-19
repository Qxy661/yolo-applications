@echo off
REM 风电场叶片缺陷检测 — 一键环境配置
REM 使用 yolo-project conda 环境

echo ========================================
echo 风电场叶片缺陷检测 - 环境配置
echo ========================================

REM 激活 conda 环境
call E:\anoconda\Scripts\activate.bat yolo-project

REM 安装依赖
echo.
echo [1/3] 安装 Python 依赖...
pip install -r requirements.txt -q

REM 验证安装
echo.
echo [2/3] 验证环境...
python -c "import ultralytics; print(f'ultralytics: {ultralytics.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

REM 下载模型权重
echo.
echo [3/3] 下载预训练权重...
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

echo.
echo ========================================
echo 环境配置完成！
echo 使用方法:
echo   conda activate yolo-project
echo   python src/train.py --model yolo11n.pt
echo ========================================
pause
