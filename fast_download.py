"""
快速多线程下载 VisDrone 数据集
"""
import os
import sys
import urllib.request
import threading
from pathlib import Path

URLS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "test": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test.zip",
}

def download_one(name, url, dest):
    dest = Path(dest)
    if dest.exists():
        sz = dest.stat().st_size / (1024**3)
        print(f"[{name}] 已存在 ({sz:.2f} GB), 跳过")
        return
    print(f"[{name}] 开始下载: {url}")
    try:
        urllib.request.urlretrieve(url, str(dest))
        sz = dest.stat().st_size / (1024**3)
        print(f"[{name}] 完成 ({sz:.2f} GB)")
    except Exception as e:
        print(f"[{name}] 失败: {e}")

def main():
    data_dir = Path(__file__).parent / "data" / "visdrone" / "zips"
    data_dir.mkdir(parents=True, exist_ok=True)

    threads = []
    for name, url in URLS.items():
        dest = data_dir / f"VisDrone2019-DET-{name}.zip"
        t = threading.Thread(target=download_one, args=(name, url, str(dest)))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("\n全部下载完成!")

if __name__ == "__main__":
    main()
