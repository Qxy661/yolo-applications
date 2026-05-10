@echo off
echo ========================================
echo   YOLO + VisDrone 环境配置
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 未安装，请先安装 Python 3.9+
    pause
    exit /b 1
)

REM 检查 NVIDIA GPU
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] 未检测到 NVIDIA GPU，将使用 CPU 训练（速度较慢）
) else (
    echo [OK] 检测到 NVIDIA GPU
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
)

echo.
echo [1/3] 安装 Python 依赖...
pip install -r requirements.txt

echo.
echo [2/3] 验证 Ultralytics 安装...
python -c "from ultralytics import YOLO; import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo.
echo [3/3] 配置完成！
echo.
echo 下一步:
echo   1. 运行 python download_data.py 下载数据集
echo   2. 运行 python src/train.py 开始训练
echo.
pause
