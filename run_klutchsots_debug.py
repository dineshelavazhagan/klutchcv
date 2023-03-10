from tracker2 import Tracker


source = "/home/roi/data/raw/IMG_3668.mp4"
classes = 32
# yolo_weights = "yolov5x.pt"
device = "0"#"cpu"#"0"
tr = Tracker(classes=classes, device=device)

tr.run_tracking(source)