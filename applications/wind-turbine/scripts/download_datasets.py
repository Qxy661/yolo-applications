"""风电场叶片缺陷检测 — 数据集下载脚本

下载公开数据集并统一为YOLO格式。

数据集来源:
1. 9类UAV风电数据集 (4,467张) — GitHub QQ767172261
2. Kaggle风电数据集 (~308张) — Kaggle
3. 7类无人机数据集 (~500张) — GitHub QQ767172261
"""
import os
import sys
import zipfile
import shutil
import yaml
from pathlib import Path
import urllib.request
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'

# 5类目标缺陷定义
TARGET_CLASSES = {
    0: 'Crack',       # 叶片裂纹
    1: 'Breakage',    # 叶片破损
    2: 'Lightning',   # 雷击损伤
    3: 'Peeling',     # 涂层脱落
    4: 'Erosion',     # 边缘侵蚀
}

# 各数据集的类别映射
DATASET_MAPPINGS = {
    '9class_uav': {
        'Chip': 1,         # Breakage
        'Crack': 0,        # Crack
        'Damaged': 1,      # Breakage
        'Damaged Or Misaligned': 1,  # Breakage
        'Discoloration': None,       # 忽略（无对应类别）
        'Erosion': 4,      # Erosion
        'Flaking': 3,      # Peeling
        'Peeling': 3,      # Peeling
        'Pinholes': None,  # 忽略
    },
    '7class_uav': {
        'Burn': 2,         # Lightning (燃烧→雷击)
        'Crack': 0,        # Crack
        'Deformation': None,
        'Dirt': None,
        'Oil Stain': None,
        'Peeling': 3,      # Peeling
        'Rust': None,
    },
    'kaggle': {
        'Crack': 0,        # Crack
        'Erosion': 4,      # Erosion
        'Broken': 1,       # Breakage
        'Hole': 1,         # Breakage
        'Leading Edge': 4, # Erosion
    },
}


def download_file(url, dest_path):
    """下载文件"""
    if dest_path.exists():
        print(f'  [SKIP] 已存在: {dest_path.name}')
        return
    print(f'  [DOWNLOAD] {url} -> {dest_path.name}')
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f'  [OK] 下载完成')
    except Exception as e:
        print(f'  [ERROR] 下载失败: {e}')


def download_9class_dataset():
    """下载9类UAV风电数据集"""
    print('\n[DATASET 1] 9类UAV风电数据集')
    dest = RAW_DIR / '9class_uav'
    dest.mkdir(parents=True, exist_ok=True)

    # 数据集来自 GitHub repo
    # 需要手动下载或使用 git clone
    readme_path = dest / 'README.md'
    if not readme_path.exists():
        print('  [INFO] 请手动下载数据集:')
        print('  URL: https://github.com/QQ767172261/Deep-Learning-YOLOV11-Model-How-to-Train-Class-9-9900-UAV-Wind-Turbine-Blade-Defect-Detection-Datase')
        print('  下载后解压到:', dest)
        print('  或使用: git clone <repo_url> ' + str(dest))
    else:
        print('  [OK] 数据集已存在')


def download_kaggle_dataset():
    """下载Kaggle风电数据集"""
    print('\n[DATASET 2] Kaggle风电数据集')
    dest = RAW_DIR / 'kaggle'
    dest.mkdir(parents=True, exist_ok=True)

    readme_path = dest / 'README.md'
    if not readme_path.exists():
        print('  [INFO] 请手动下载数据集:')
        print('  搜索: https://www.kaggle.com/search?q=wind+turbine+blade+defect')
        print('  下载后解压到:', dest)
    else:
        print('  [OK] 数据集已存在')


def download_7class_dataset():
    """下载7类无人机数据集"""
    print('\n[DATASET 3] 7类无人机数据集')
    dest = RAW_DIR / '7class_uav'
    dest.mkdir(parents=True, exist_ok=True)

    readme_path = dest / 'README.md'
    if not readme_path.exists():
        print('  [INFO] 请手动下载数据集:')
        print('  URL: https://github.com/QQ767172261/Deep-Learning-YOLOv8-Model-Training-UAV-Aerial-Wind-Turbine-Blade-Defect-Detection-Dataset-Detection')
        print('  下载后解压到:', dest)
    else:
        print('  [OK] 数据集已存在')


def main():
    print('=' * 60)
    print('风电场叶片缺陷检测 — 数据集下载')
    print('=' * 60)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    download_9class_dataset()
    download_kaggle_dataset()
    download_7class_dataset()

    print('\n' + '=' * 60)
    print('下载完成！')
    print('请将数据集解压到对应的 raw/ 目录后，运行 merge_datasets.py 进行合并。')
    print('=' * 60)


if __name__ == '__main__':
    main()
