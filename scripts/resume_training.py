"""Resume P2+CBAM training from last.pt with logging to file."""
import sys, os
from pathlib import Path
from datetime import datetime

def main():
    # Setup logging
    log_path = Path(__file__).parent.parent / "runs" / "p2" / "yolov8s_p2_cbam" / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()

    log_file = open(log_path, "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    print("\n" + "=" * 60)
    print("[%s] Resuming P2+CBAM training" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    import PIL.Image
    _pil_open = PIL.Image.open

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    import register_custom_modules

    from ultralytics import YOLO

    PIL.Image.open = _pil_open

    project_root = Path(__file__).parent.parent
    last_pt = project_root / "runs" / "p2" / "yolov8s_p2_cbam" / "weights" / "last.pt"

    if not last_pt.exists():
        print("[ERROR] last.pt not found at", last_pt)
        sys.exit(1)

    print("[INFO] Resuming from: %s" % last_pt)

    model = YOLO(str(last_pt))
    results = model.train(
        resume=True,
        data=str(project_root / "data" / "visdrone" / "visdrone.yaml"),
        batch=3,
        workers=2,
    )

    print("\nTraining complete!")
    print("mAP@0.5: %.4f" % results.box.map50)
    print("mAP@0.5:0.95: %.4f" % results.box.map)
    log_file.close()

if __name__ == "__main__":
    main()
