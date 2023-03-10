from tracker import Tracker,Suported_resolutions

classes = 0
zoom_factor=1
# yolo_weights = "yolov5x.pt"
device = "cuda:0"#
tr = Tracker(classes=classes, device=device)
file_name = 'IMG_3853_2_cropped_high.mp4'
source = "../data/raw/IMG_3853_2.mp4"
tr.run_tracking_per_frame(source)
frame_size = Suported_resolutions['high']
new_vid = tr.create_new_video(frame_size,zoom_factor,file_name)
