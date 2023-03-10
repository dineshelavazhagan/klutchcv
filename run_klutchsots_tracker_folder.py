from tracker import Tracker,Suported_resolutions
import os
# from google.colab import auth

# auth.authenticate_user()
# project_id = 'nodal-clock-365917'
# # !gcloud config set project {project_id}
# # !gsutil ls

classes = 32
zoom_factor=1
# yolo_weights = "yolov5x.pt"
device = "cpu"#
tr = Tracker(classes=classes, device=device)
folder='all_videos'
frame_size = Suported_resolutions['high']

root_folder = f"../data/raw/{folder}"
output_folder = f"../data/proccessed/{folder}"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

files = os.listdir(root_folder)
for file in files:
    source = os.path.join(root_folder,file)
    print('Working on',source)
    file_name = os.path.join(output_folder,file)
    tr.run_tracking_per_frame(source)
    new_vid = tr.create_new_video(frame_size,zoom_factor,file_name)
