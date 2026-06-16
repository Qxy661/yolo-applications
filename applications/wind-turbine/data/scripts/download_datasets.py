"""
风电叶片缺陷检测数据集下载脚本
================================
数据集来源：
1. zhaowenhai2023 - 高分辨率表面缺陷数据集，3,800+张 (百度网盘)
2. Blade30 - 真实无人机巡检数据，1,302张
3. QQ767172261 六类数据集 - 3,282张
4. QQ767172261 UAV五类数据集 - 4,467张
5. WTBD数据集 - Nature Scientific Data 2026 (如能获取)

使用方法：
    python download_datasets.py --all          # 下载所有数据集
    python download_datasets.py --zhaowenhai   # 只下载zhaowenhai2023
    python download_datasets.py --blade30      # 只下载Blade30
    python download_datasets.py --qq6          # 只下载QQ六类
    python download_datasets.py --qq5          # 只下载QQ UAV五类
"""

import os
import sys
import argparse
import subprocess
import zipfile
import shutil
from pathlib import Path

# 路径配置
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / 'raw'
ZHAOWENHAI_DIR = RAW_DIR / 'zhaowenhai2023'
BLADE30_DIR = RAW_DIR / 'blade30'
QQ6_DIR = RAW_DIR / 'qq767172261_6cls'
QQ5_DIR = RAW_DIR / 'qq767172261_uav5'
WTBD_DIR = RAW_DIR / 'wtbd'


def check_dependencies():
    """检查依赖"""
    try:
        import gdown
        print("[OK] gdown 已安装")
    except ImportError:
        print("[INFO] gdown 未安装，将使用手动下载方式")

    try:
        import requests
        print("[OK] requests 已安装")
    except ImportError:
        print("[INFO] requests 未安装，将使用手动下载方式")


def download_zhaowenhai2023():
    """
    下载 zhaowenhai2023 高分辨率表面缺陷数据集
    来源: https://github.com/zhaowenhai2023/Wind-turbine-blade-surface-defect-dataset
    数量: 3,800+ 张高分辨率图片
    特点: 使用StyleGAN3和PBGM方法增强，背景多样
    """
    print("\n" + "="*60)
    print("下载 zhaowenhai2023 高分辨率表面缺陷数据集")
    print("="*60)

    ZHAOWENHAI_DIR.mkdir(parents=True, exist_ok=True)

    # 百度网盘链接
    baidu_links = [
        ("https://pan.baidu.com/s/1_tJBlhuNE1eDMxTO9uzdWA", "1234", "图片数据"),
        ("https://pan.baidu.com/s/1Qr0jRVLFUpa13HH0HDsC5Q", "1234", "标注数据"),
    ]

    print("\nzhaowenhai2023 数据集下载方式：")
    print("-" * 40)
    print("方式1: 百度网盘 (推荐，国内速度快)")
    for link, code, desc in baidu_links:
        print(f"  {desc}: {link} (码:{code})")
    print()

    print("\n[TODO] 请手动完成以下步骤：")
    print("1. 从百度网盘下载两个压缩包到:", ZHAOWENHAI_DIR)
    print("2. 解压到:", ZHAOWENHAI_DIR)
    print("3. 目录结构应为:")
    print("   zhaowenhai2023/")
    print("   ├── images/")
    print("   │   ├── img_001.jpg")
    print("   │   └── ...")
    print("   └── annotations/")
    print("       ├── img_001.txt")
    print("       └── ...")

    print("\n数据集信息：")
    print("- 图片总数: 3,800+ 张")
    print("- 图片类型: 高分辨率叶片表面缺陷")
    print("- 增强方法: StyleGAN3 + PBGM")
    print("- 背景: 多样化")


def download_blade30():
    """
    下载 Blade30 数据集
    来源: https://github.com/cong-yang/Blade30
    发表: Renewable Energy, Vol.203, 2023
    """
    print("\n" + "="*60)
    print("下载 Blade30 数据集")
    print("="*60)

    BLADE30_DIR.mkdir(parents=True, exist_ok=True)

    # 百度网盘链接 (需要手动下载)
    baidu_links = [
        ("https://pan.baidu.com/s/17kv5Xadz1QcSrvoG58WtBw", "码:1234", "叶片1-15"),
        ("https://pan.baidu.com/s/1hzcwdc6sBXOeja3nkfartg", "码:1234", "叶片16-30"),
    ]

    # Google Drive 链接 (可尝试自动下载)
    gdrive_links = [
        ("1HbB4t9xV2oCgSSxR9hMEOU6v9qDfetmR", "叶片1-15"),
        ("1SwRdMzA7zCkNVlHuWvk8uK6eDToM0mUV", "叶片16-30"),
    ]

    print("\nBlade30 数据集下载方式：")
    print("-" * 40)
    print("方式1: 百度网盘 (推荐，国内速度快)")
    for link, code, desc in baidu_links:
        print(f"  {desc}: {link} ({code})")
    print()
    print("方式2: Google Drive")
    for file_id, desc in gdrive_links:
        print(f"  {desc}: https://drive.google.com/uc?id={file_id}")
    print()
    print("方式3: OneDrive")
    print("  https://1drv.ms/u/s!AoXJBmXKVWu5tmtUzCJULhrtYuIP")
    print()

    # 尝试用 gdown 下载 Google Drive 文件
    try:
        import gdown
        print("[INFO] 尝试从 Google Drive 下载...")
        for file_id, desc in gdrive_links:
            output = BLADE30_DIR / f"blade30_{desc}.zip"
            if output.exists():
                print(f"  [SKIP] {desc} 已存在")
                continue
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, str(output), quiet=False)
                print(f"  [OK] {desc} 下载完成")
            except Exception as e:
                print(f"  [WARN] {desc} 下载失败: {e}")
                print(f"  请手动下载: https://drive.google.com/uc?id={file_id}")
    except ImportError:
        print("[INFO] gdown 未安装，请手动下载")
        print("请手动下载百度网盘文件到:", BLADE30_DIR)
    except Exception as e:
        print(f"[WARN] Google Drive 下载失败: {e}")
        print("请手动下载百度网盘文件到:", BLADE30_DIR)

    print("\n[TODO] 请手动完成以下步骤：")
    print("1. 从百度网盘下载两个压缩包到:", BLADE30_DIR)
    print("2. 解压到:", BLADE30_DIR)
    print("3. 目录结构应为:")
    print("   blade30/")
    print("   ├── images/")
    print("   │   ├── blade_001/")
    print("   │   └── ...")
    print("   └── annotations/")
    print("       ├── blade_001.json")
    print("       └── ...")


def download_qq767172261_6cls():
    """
    下载 QQ767172261 六类叶片缺陷数据集
    来源: https://github.com/QQ767172261/...
    类别: Crack, Erosion, Dirt, Oil leakage, PU-tape, Pin Hole
    数量: 3,282张
    """
    print("\n" + "="*60)
    print("下载 QQ767172261 六类数据集")
    print("="*60)

    QQ6_DIR.mkdir(parents=True, exist_ok=True)

    # GitHub 仓库链接
    repo_url = "https://github.com/QQ767172261/Deep-Learning-How-the-YOLOV8-Model-Trains-Wind-Turbine-Blade-Defect-Detection-Datasets-Establish-Dee"

    print(f"\nGitHub 仓库: {repo_url}")
    print("\n注意: 该仓库可能只包含 README，实际数据需要从其他平台获取")
    print("\n尝试克隆仓库...")

    try:
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, str(QQ6_DIR / 'repo')],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print("[OK] 仓库克隆成功")
            # 检查是否有数据文件
            data_files = list((QQ6_DIR / 'repo').rglob('*.txt'))
            if data_files:
                print(f"[OK] 找到 {len(data_files)} 个标注文件")
            else:
                print("[INFO] 仓库中未找到数据文件，需要从其他平台下载")
        else:
            print(f"[WARN] 克隆失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("[WARN] 克隆超时")
    except Exception as e:
        print(f"[WARN] 克隆失败: {e}")

    print("\n[TODO] 如果仓库中没有数据，请尝试以下方式获取：")
    print("1. 在 GitHub Issues 中联系作者")
    print("2. 搜索中国AI数据平台 (如 AI Studio, 魔搭社区)")
    print("3. 搜索关键词: '风电叶片缺陷检测数据集 YOLO'")

    print("\n数据集信息：")
    print("- 图片总数: 3,282张")
    print("- 训练集: 2,743 / 验证集: 270 / 测试集: 269")
    print("- 类别分布:")
    print("  Crack: 1,144张 (2,163个框)")
    print("  Erosion: 233张 (337个框)")
    print("  Dirt: 693张 (762个框)")
    print("  Oil leakage: 574张 (600个框)")
    print("  PU-tape: 329张 (621个框)")
    print("  Pin Hole: 303张 (303个框)")


def download_qq767172261_uav5():
    """
    下载 QQ767172261 UAV航拍五类缺陷数据集
    来源: https://github.com/QQ767172261/...
    类别: Oil Leakage, Dirt, Paint (Peeling), LE-Erosion, PU-tape
    数量: 4,467张
    """
    print("\n" + "="*60)
    print("下载 QQ767172261 UAV五类数据集")
    print("="*60)

    QQ5_DIR.mkdir(parents=True, exist_ok=True)

    # GitHub 仓库链接
    repo_url = "https://github.com/QQ767172261/Deep-Learning-YOLOv8-Model-Training-UAV-Aerial-Wind-Turbine-Blade-Defect-Detection-Dataset-Detection"

    print(f"\nGitHub 仓库: {repo_url}")
    print("\n注意: 该仓库可能只包含 README，实际数据需要从其他平台获取")
    print("\n尝试克隆仓库...")

    try:
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', repo_url, str(QQ5_DIR / 'repo')],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print("[OK] 仓库克隆成功")
            # 检查是否有数据文件
            data_files = list((QQ5_DIR / 'repo').rglob('*.txt'))
            if data_files:
                print(f"[OK] 找到 {len(data_files)} 个标注文件")
            else:
                print("[INFO] 仓库中未找到数据文件，需要从其他平台下载")
        else:
            print(f"[WARN] 克隆失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("[WARN] 克隆超时")
    except Exception as e:
        print(f"[WARN] 克隆失败: {e}")

    print("\n[TODO] 如果仓库中没有数据，请尝试以下方式获取：")
    print("1. 在 GitHub Issues 中联系作者")
    print("2. 搜索中国AI数据平台")
    print("3. 搜索关键词: 'UAV 风电叶片缺陷 YOLO'")

    print("\n数据集信息：")
    print("- 图片总数: 4,467张")
    print("- 训练集: 3,898 / 验证集: 189 / 测试集: 380")
    print("- 类别分布:")
    print("  Oil Leakage: 753个框")
    print("  Dirt: 846个框")
    print("  Paint (Peeling): 2,455个框")
    print("  LE-Erosion: 617个框")
    print("  PU-tape: 700个框")


def download_wtbd():
    """
    尝试获取 WTBD 数据集
    来源: Nature Scientific Data 2026
    论文: https://www.nature.com/articles/s41597-026-06762-x
    """
    print("\n" + "="*60)
    print("获取 WTBD 数据集 (Nature Scientific Data 2026)")
    print("="*60)

    WTBD_DIR.mkdir(parents=True, exist_ok=True)

    paper_url = "https://www.nature.com/articles/s41597-026-06762-x"
    print(f"\n论文链接: {paper_url}")
    print("\nNature Scientific Data 数据集通常在论文 Data Availability 章节提供下载链接")
    print("\n[TODO] 请手动完成以下步骤：")
    print("1. 访问论文页面")
    print("2. 查找 Data Availability 或 Data Citations 部分")
    print("3. 下载数据集到:", WTBD_DIR)
    print("4. 常见下载方式:")
    print("   - figshare.com")
    print("   - zenodo.org")
    print("   - GitHub 仓库")


def main():
    parser = argparse.ArgumentParser(description='下载风电叶片缺陷检测数据集')
    parser.add_argument('--all', action='store_true', help='下载所有数据集')
    parser.add_argument('--zhaowenhai', action='store_true', help='只下载zhaowenhai2023高分辨率数据集')
    parser.add_argument('--blade30', action='store_true', help='只下载Blade30')
    parser.add_argument('--qq6', action='store_true', help='只下载QQ六类数据集')
    parser.add_argument('--qq5', action='store_true', help='只下载QQ UAV五类数据集')
    parser.add_argument('--wtbd', action='store_true', help='获取WTBD数据集信息')
    args = parser.parse_args()

    # 如果没有指定任何参数，显示帮助
    if not any([args.all, args.zhaowenhai, args.blade30, args.qq6, args.qq5, args.wtbd]):
        parser.print_help()
        return

    print("="*60)
    print("风电叶片缺陷检测数据集下载")
    print("="*60)
    print(f"目标目录: {RAW_DIR}")

    # 检查依赖
    check_dependencies()

    # 下载数据集
    if args.all or args.zhaowenhai:
        download_zhaowenhai2023()

    if args.all or args.blade30:
        download_blade30()

    if args.all or args.qq6:
        download_qq767172261_6cls()

    if args.all or args.qq5:
        download_qq767172261_uav5()

    if args.all or args.wtbd:
        download_wtbd()

    # 总结
    print("\n" + "="*60)
    print("下载总结")
    print("="*60)
    print(f"zhaowenhai2023 目录: {ZHAOWENHAI_DIR}")
    print(f"Blade30 目录: {BLADE30_DIR}")
    print(f"QQ六类目录: {QQ6_DIR}")
    print(f"QQ五类目录: {QQ5_DIR}")
    print(f"WTBD目录: {WTBD_DIR}")
    print("\n下一步: 运行数据清洗和格式转换脚本")


if __name__ == '__main__':
    main()
