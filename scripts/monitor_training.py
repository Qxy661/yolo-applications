"""Monitor P2+CBAM training - accurate progress via reading file tail."""
import time, os, re

log_path = r"E:\yolo-visdrone\runs\p2\yolov8s_p2_cbam\training.log"
csv_path = r"E:\yolo-visdrone\runs\p2\yolov8s_p2_cbam\results.csv"
TOTAL_EPOCHS = 40

def get_latest_progress():
    try:
        size = os.path.getsize(log_path)
        with open(log_path, 'rb') as f:
            f.seek(max(0, size - 10000))
            data = f.read().decode('utf-8', errors='ignore')
        lines = data.split('\r')
        for line in reversed(lines):
            m = re.search(r'(\d+)/80\s+([\d.]+)G\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+640:\s+(\d+)%.*?(\d+)/(\d+)\s', line)
            if m:
                return {
                    'epoch': int(m.group(1)),
                    'mem': m.group(2),
                    'box': float(m.group(3)),
                    'cls': float(m.group(4)),
                    'dfl': float(m.group(5)),
                    'pct': int(m.group(6)),
                    'iter': int(m.group(7)),
                    'total': int(m.group(8)),
                }
    except:
        pass
    return None

def get_completed():
    try:
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        if len(lines) > 2:
            last = lines[-1].strip().split(',')
            return len(lines) - 2, float(last[7])
    except:
        pass
    return 0, 0

last_reported = ""
while True:
    try:
        p = get_latest_progress()
        if p:
            completed, last_map = get_completed()
            bar_len = 30
            filled = bar_len * p['pct'] // 100
            bar = '#' * filled + '-' * (bar_len - filled)
            key = "%d-%d" % (p['epoch'], p['iter'])
            if key != last_reported:
                last_reported = key
                eta_min = (p['total'] - p['iter']) / 8.0 / 60
                print("Epoch %d/%d [%s] %d/%d (%d%%) ETA %.1fm | box=%.3f cls=%.3f dfl=%.3f | done=%d mAP=%.4f" % (
                    p['epoch'], TOTAL_EPOCHS, bar, p['iter'], p['total'], p['pct'],
                    eta_min, p['box'], p['cls'], p['dfl'], completed, last_map), flush=True)
                if completed >= TOTAL_EPOCHS:
                    print("Training complete!", flush=True)
                    break
    except Exception as e:
        print("Error: %s" % e, flush=True)
    time.sleep(30)
