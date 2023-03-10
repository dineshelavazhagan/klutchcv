import argparse
import cv2

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

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

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from trackers.multi_tracker_zoo import create_tracker

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])

Suported_resolutions = {
    'very_low':[200,320],
    'low':[360,640],
    'mid':[640,960],
    'high':[720,1280]
}#xy

@torch.no_grad()
class Tracker():
    def __init__(self,
        source='0',
        yolo_weights=WEIGHTS / 'best.pt',#'yolov5x.pt',  # model.pt path(s),
        reid_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cuda:0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        eval=False,  # run multi-gpu eval
        debug=False,
        skip_frames=1,#how many frames to skip between detections
        ):

        # Directories
        if not isinstance(yolo_weights, list):  # single yolo model
            exp_name = yolo_weights.stem
        elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
            exp_name = Path(yolo_weights[0]).stem
        else:  # multiple models after --yolo_weights
            exp_name = 'ensemble'
        exp_name = name if name else exp_name + "_" + reid_weights.stem
        save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
        (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        self.hide_class = hide_class
        self.hide_conf = hide_conf
        self.hide_labels = hide_labels
        self.save_vid = save_vid
        self.save_dir = save_dir
        self.nosave = nosave
        self.tracking_method = tracking_method
        self.reid_weights = reid_weights
        self.device = device
        self.half = half
        self.augment = augment
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det=max_det
        self.save_crop = save_crop
        self.line_thickness = line_thickness
        self.save_txt = save_txt
        self.visualize = visualize
        self.update = update
        self.skip_frames = skip_frames
        self.debug = debug
        # Load model
        self.load_model(eval,yolo_weights,dnn,half,imgsz)
    
    @torch.no_grad()
    def run_tracking_per_frame(self,source):
        source = str(source)
        save_img = not self.nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        show_vid = True
            
        if is_url and is_file:
            source = check_file(source)  # download
        # Dataloader
        if webcam:
            show_vid = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(source, img_size=self.imgsz, stride=self.model.stride, auto=self.model.pt)
            nr_sources = len(self.dataset)
        else:
            self.dataset = LoadImages(source, img_size=self.imgsz, stride=self.model.stride, auto=self.model.pt)
            nr_sources = 1
        self.source = source
        vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

        # Create as many strong sort instances as there are video sources
        tracker_list = []
        for i in range(nr_sources):
            # print("tracker", self.device)
            tracker = create_tracker(self.tracking_method, self.reid_weights, self.device, self.half)
            tracker_list.append(tracker, )
            if hasattr(tracker_list[i], 'model'):
                if hasattr(tracker_list[i].model, 'warmup'):
                    tracker_list[i].model.warmup()
        outputs = [None] * nr_sources

        # Run tracking
        self.model.warmup(imgsz=(1 if self.model.pt else nr_sources, 3, *self.imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
        centers = np.zeros([self.dataset.frames,2])
        print(centers.shape)
        if self.debug:
            detections = np.zeros([self.dataset.frames,4])
        # ddims = np.zeros([self.dataset.frames,1])
        # ratios = np.zeros([self.dataset.frames,1])
        # for each in self.dataset:
        #     print(type(each))
            
        print("------------broken----------------------", torch.cuda.is_available())
        for frame_idx, (path, im, im0s, vid_cap, s_ddim_ratio) in enumerate(self.dataset):
            # print(s_ddim_ratio, s_ddim_ratio.split(" "))
            # s,ddim,ratio = s_ddim_ratio.split(" ")
            s = s_ddim_ratio
            if frame_idx%self.skip_frames!=0:
                continue
            t1 = time_sync()
            self.device = "cuda:0"
            im = torch.from_numpy(im)#.to(device)
            im = im.half() if self.half else im.float()  # uint8 to fp16/32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            # print(self.device)
            im = im.to(self.device)
            t2 = time_sync()
            dt[0] += t2 - t1
            print("pre_process", t2 - t1)

            # Inference
            visualize = increment_path(self.save_dir / Path(path[0]).stem, mkdir=True) if self.visualize else False
            pred = self.model(im, augment=self.augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2
            print("inference", t3 - t2)

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
            dt[2] += time_sync() - t3
            print("nms", time_sync() - t3)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if len(det) == 0:
                    continue
                if int(det[0][5].item()) != self.classes:
                    continue
                # print("here")

                seen += 1
                if webcam:  # nr_sources >= 1
                    p, im0, _ = path[i], im0s[i].copy(), self.dataset.count
                    p = Path(p)  # to Path
                    s += f'{i}: '
                    txt_file_name = p.name
                    save_path = str(self.save_dir / p.name)  # im.jpg, vid.mp4, ...
                else:
                    p, im0, _ = path, im0s.copy(), getattr(self.dataset, 'frame', 0)
                    p = Path(p)  # to Path
                    # video file
                    if source.endswith(VID_FORMATS):
                        txt_file_name = p.stem
                        save_path = str(self.save_dir / p.name)  # im.jpg, vid.mp4, ...
                    # folder with imgs
                    else:
                        txt_file_name = p.parent.name  # get folder name containing current img
                        save_path = str(self.save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
                curr_frames[i] = im0

                txt_path = str(self.save_dir / 'tracks' / txt_file_name)  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                imc = im0.copy() if self.save_crop else im0  # for save_crop

                annotator = Annotator(im0, line_width=self.line_thickness, pil=not ascii)
                #if cfg.STRONGSORT.ECC:  # camera motion compensation
                #    strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()  # xyxy
                    cnt = find_center(det[0, :4])
                    centers[frame_idx,:] = cnt
                    if self.debug:
                        detections[frame_idx,:] = det[0, :4]
                    # bboxs[frame_idx,:] = det[0, :4]
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{c}:{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # pass detections to strongsort
                    t4 = time_sync()
                    outputs[i] = tracker_list[i].update(det.cpu(), im0)
                    t5 = time_sync()
                    dt[3] += t5 - t4

                    # draw boxes for visualization
                    if len(outputs[i]) > 0:
                        for j, (output, conf) in enumerate(zip(outputs[i], det[:, 4])):
        
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]

                            if self.save_txt:
                                # to MOT format
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2] - output[0]
                                bbox_h = output[3] - output[1]
                                # Write MOT compliant results to file
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                            if self.save_vid or self.save_crop or show_vid:  # Add bbox to image
                                c = int(cls)  # integer class
                                id = int(id)  # integer id
                                label = None if self.hide_labels else (f'{id} {self.model.names[c]}' if self.hide_conf else \
                                    (f'{id} {conf:.2f}' if self.hide_class else f'{id} {self.model.names[c]} {conf:.2f}'))
                                annotator.box_label(bboxes, label, color=colors(c, True))
                                # print("annotion done")
                                if self.save_crop:
                                    txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                    save_one_box(bboxes, imc, file=self.save_dir / 'crops' / txt_file_name / self.model.names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                    # LOGGER.info(f'{s}Done. yolo:({t3 - t2:.3f}s), {tracking_method}:({t5 - t4:.3f}s)')
                    LOGGER.info(f'{s}Done.')

                else:
                    #strongsort_list[i].increment_ages()
                    LOGGER.info('No detections')

                
                # Stream results
                im0 = annotator.result()
                if show_vid:
                    show_vio = cv2.resize(im0, (int(im0.shape[1]/4), int(im0.shape[0]/ 4)))
                    cv2.imshow(str(p), show_vio)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if self.save_vid:
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 15, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

                prev_frames[i] = curr_frames[i]


        # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {self.tracking_method} update per image at shape {(1, 3, *self.imgsz)}' % t)
        if self.save_txt or self.save_vid:
            s = f"\n{len(list(self.save_dir.glob('tracks/*.txt')))} tracks saved to {self.save_dir / 'tracks'}" if self.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        if self.update:
            strip_optimizer(self.yolo_weights)  # update model (to fix SourceChangeWarning)

        # interpolate centers
        self.interpolate_empty_values(centers)

        self.centers = centers

    def create_new_video(self,frame_size,zoom_factor,file_name):
        full_path = os.path.join("../data/proccessed",file_name)
        temp_full_path = "../data/proccessed/temp.mp4"
        if os.path.exists(full_path):
            os.remove(full_path)
        if os.path.exists(temp_full_path):
            os.remove(temp_full_path)
        out = cv2.VideoWriter(temp_full_path, cv2.VideoWriter_fourcc(*'XVID'), 50, frame_size)         
        dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.model.stride, auto=self.model.pt)
        for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
            if frame_idx%30==0:
                print('writeing frame',frame_idx)
            ccrop_img = self.center_crop(im0s,self.centers[frame_idx],frame_size)
            zoom_im = self.zoom_image(ccrop_img, factor=zoom_factor)
            out.write(zoom_im)
    
        out.release()
        cv2.destroyAllWindows()
        print('Mapping video and audio')
        os.system(f"ffmpeg -i {temp_full_path} -i {self.source} -c copy -map 0:v:0 -map 1:a:0 -shortest {full_path}")
        os.remove(temp_full_path)
        print('finished creating video',file_name)
        
    def center_crop(self,img, center, dim):
        """Returns center cropped image
        Args:
        img: image to be center cropped
        dim: dimensions (width, height) to be cropped
        """
        width, height = img.shape[1], img.shape[0]

        # process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<width else width
        crop_height = dim[1] if dim[1]<height else height 

        #update 
        mid_x, mid_y = int(center[0]),int(center[1])#int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2)

        #handle widht
        if mid_x + cw2 > width and mid_x - cw2 > 0:
            mid_x = width - cw2
        elif mid_x - cw2 < 0 and mid_x + cw2 < width:
            mid_x = cw2
        else:
            pass
        
        #handle height
        if mid_y - ch2 < 0 and mid_y + ch2 < height:
            mid_y = ch2
        elif mid_y + ch2 > height and mid_y - ch2 > 0:
            mid_y = height - ch2
        else:
            pass
            
        crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2,:]
        return crop_img

    def zoom_image(self,img, factor=1):
        """Returns resize image by scale factor.
        This helps to retain resolution ratio while resizing.
        Args:
        img: image to be scaled
        factor: scale factor to resize
        """
        new_img = cv2.resize(img,(int(img.shape[1]*factor), int(img.shape[0]*factor)))
        width, height = img.shape[1], img.shape[0]
        new_width, new_height = new_img.shape[1], new_img.shape[0]
        if width<new_width and height<new_height:
            cw2, ch2 = int(width/2), int(height/2)
            cntx,cnty = int(new_width/2), int(new_height/2)
            return new_img[cnty-ch2:cnty+ch2,cntx-cw2:cntx+cw2,:]
        else:
            return img


    def interpolate_empty_values(self,centers):
        detect_indecies = np.nonzero(centers[:,0])[0]
        
        #handle all the values
        #TODO handle issue where the first frame is not detected
        for idx,curr_idx in enumerate(detect_indecies):
            if idx == 0:
                centers[:detect_indecies[0],0] = [centers[detect_indecies[0],0]]*len(centers[:detect_indecies[0],0])
                centers[:detect_indecies[0],1] = [centers[detect_indecies[0],1]]*len(centers[:detect_indecies[0],1])
                continue

            prev_idx = detect_indecies[idx-1]
            # curr_idx = detect_indecies[idx]

            #handle x
            centers[prev_idx:curr_idx,0] = np.linspace(centers[prev_idx,0],centers[curr_idx,0],curr_idx-prev_idx+1)[:-1]
            #handle y
            centers[prev_idx:curr_idx,1] = np.linspace(centers[prev_idx,1],centers[curr_idx,1],curr_idx-prev_idx+1)[:-1]

        # handle last value
        centers[centers[:,0]==0,0] = [centers[detect_indecies[-1],0]]*(len(centers) - detect_indecies[-1]-1)
        centers[centers[:,1]==0,1] = [centers[detect_indecies[-1],1]]*(len(centers) - detect_indecies[-1]-1)

    def load_model(self,eval,yolo_weights,dnn,half,imgsz):
        if eval:
                self.device = torch.device(int(self.device))
        else:
            self.device =torch.device(self.device)# select_device(device)
        self.device=torch.device("cuda")
        self.model = DetectMultiBackend(yolo_weights, device=self.device, dnn=dnn, data=None, fp16=half)
        # stride, names, pt = model.stride, model.names, model.pt
        self.imgsz = check_img_size(imgsz, s=self.model.stride)  # check image size


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--reid-weights', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--tracking-method', type=str, default='strongsort', help='strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')  
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--eval', action='store_true', help='run evaluation')
    parser.add_argument('--debug', action='store_true', help='run evaluation')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def find_center(coords):
    x1,y1,x2,y2 = coords
    
    x = (x2.item()+x1.item())/2
    y = (y2.item()+y1.item())/2
    return x,y

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2
 

def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
