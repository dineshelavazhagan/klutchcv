import argparse
import os
import platform
import sys
from pathlib import Path

import torch
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'ocsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'ocsort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, Profile,non_max_suppression, scale_boxes,check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
# from trackers.multi_tracker_zoo import create_tracker
from yolov5.utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)
import numpy as np
# Load model

class detect_person:
    def __init__(self):
        self.weights= WEIGHTS / 'yolov5x.pt'
        self.device = ''
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=None, fp16=False)
        self.stride, self.names, pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size((640, 640), s=self.stride)  # check image size
        self.save_dir = "./"
        self.model.warmup(imgsz=(1 if pt or self.model.triton else 1, 3, *self.imgsz))  # warmup
        self.seen, windows, self.dt = 0, [], (Profile(), Profile(), Profile())
        self.conf_thres=0.55
        self.iou_thres=0.45
        self.classes=32
        self.agnostic_nms=False
        self.max_det=1000
        self.hide_labels = False
        self.hide_conf = False
    # for path, im, im0s, vid_cap, s in dataset:

    def infer(self,im):
        # im = cv2.imread("stats_sam/break_digi/463_463786670_7666917.png")
        im0s = im.copy()
        im = letterbox(im0s, 640, stride=self.stride, auto=True)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = self.model(im, augment=False)

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            self.seen += 1
            im0 = im0s.copy()
            # p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # im.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=1, example=str(self.names))
            # print("names------------->", self.names)
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                # print(det[:, :4])
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # print(det[0])
                det = sorted(det, key=lambda x: x[0])
                
                # Write results
                for chars_count, (*xyxy, conf, cls) in enumerate(det):
                    # if save_txt:  # Write to file
                    # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    # line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    # with open(f'{txt_path}.txt', 'a') as f:
                    #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    # if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')
                    # self.all_locations.append([x+0,y+0, x+int(chars.shape[1]/2), y+h+chars.shape[0]])
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # if save_crop:
                        # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            im0 = annotator.result()
            # if view_img:
            #     if platform.system() == 'Linux' and p not in windows:
            #         windows.append(p)
            #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            # print(self.all_ticks)
            show_vio = cv2.resize(im0, (int(im0.shape[1]/4), int(im0.shape[0]/ 4)))
            cv2.imshow(str("imshoe"), show_vio)
            cv2.waitKey(1)  # 1 millisecond
            # cv2.imshow("str(p)", im0)
            # cv2.waitKey(0)  # 1 millisecond

detect = detect_person()
# # path = "stats_sam/break_digi/463_463786670_7666917.png"
# path = "results/image_logs/636/1/3.png"
# im = cv2.imread(path)


vs = cv2.VideoCapture("../data/raw/IMG_3853_2.mp4")

while True:
    _, frame = vs.read()
    # cv2.imshow("frame", frame)
    # cv2.waitKey(0)
    detect.infer(frame)