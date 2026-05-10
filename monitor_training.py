"""
训练监控脚本 - 安全守护
检查磁盘空间和训练状态
"""
import shutil
import time
import os
import sys

def check_safety():
    c_free = shutil.disk_usage('C:/').free // 2**30
    e_free = shutil.disk_usage('E:/').free // 2**30
    ts = time.strftime('%H:%M:%S')

    if c_free < 5:
        print('[%s] DANGER: C: only %dGB free! STOPPING!' % (ts, c_free))
        return False
    if e_free < 10:
        print('[%s] DANGER: E: only %dGB free! STOPPING!' % (ts, e_free))
        return False

    print('[%s] SAFE: C: %dGB, E: %dGB' % (ts, c_free, e_free))
    return True

def check_training():
    results_csv = r'E:\yolo-visdrone\runs\detect\runs\baseline\yolov8n_visdrone\results.csv'
    best_pt = r'E:\yolo-visdrone\runs\detect\runs\baseline\yolov8n_visdrone\weights\best.pt'

    if os.path.exists(best_pt):
        print('TRAINING_COMPLETE: best.pt found!')
        return True

    if os.path.exists(results_csv):
        with open(results_csv, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last = lines[-1].strip().split(',')
                epoch = last[0]
                map50 = last[6] if len(last) > 6 else 'N/A'
                print('Epoch %s, mAP@0.5: %s' % (epoch, map50))

    return False

if __name__ == '__main__':
    print('Training monitor started. Checking every 2 minutes...')
    print('Safety thresholds: C: < 5GB or E: < 10GB = STOP')
    print()

    while True:
        if not check_safety():
            print('SAFETY ALERT! Please check disk space!')
            sys.exit(1)

        if check_training():
            print('Training finished!')
            sys.exit(0)

        time.sleep(120)
