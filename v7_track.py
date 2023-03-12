import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

print(torch.cuda.is_available())

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'



import os
import numpy as np



import sys,os
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'ocsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'ocsort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadStreams, LoadImages
from yolov7.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel



class ball_det():
    def __init__(self):
        self.Suported_resolutions = {
                'very_low':[200,320],
                'low':[360,640],
                'mid':[640,960],
                'high':[720,1280]
            }#xy
        self.imgsz=(640, 640)
        self.perv_center = (-1,-1)
        self.weights='yolov7x.pt' 
        self.img_size=640
        self.conf_thres=0.55
        self.iou_thres=0.45
        self.device='cuda'
        self.view_img=False
        self.save_txt=False
        self.save_conf=False
        self.nosave=False
        self.classes=32
        self.agnostic_nms=False
        self.augment=False
        self.update=False
        self.project='runs/detect'
        self.name='exp'
        self.exist_ok=False
        self.no_trace=False
        

    def find_center(self,coords):
        short_x = []
        short_y = []
        al_x = []
        al_y = []
        for coord in coords:
            x1,y1,x2,y2 = coord
            x = (x2.item()+x1.item())/2
            y = (y2.item()+y1.item())/2
            short_x.append(abs(self.perv_center[0] - x)) 
            short_y.append(abs(self.perv_center[1] - y))
            al_x.append(x)
            al_y.append(y)
        x = al_x[short_x.index(min(short_x))]
        y = al_y[short_y.index(min(short_y))]
        # self.prev_x = x
        # self.prev_y = y
        self.perv_center  =(x,y)
        print(x,y)

        return x,y

    def detect(self,source,save_img=False):
        weights,view_img, save_txt, self.imgsz, trace = self.weights, self.view_img, self.save_txt, self.img_size, not self.no_trace
        save_img = not self.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(self.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        self.imgsz = check_img_size(self.imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, self.img_size)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=self.imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=self.imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        t0 = time.time()
        # tracker_list = []
        # nr_sources = len(dataset)
        # for i in range(nr_sources):
        #     # print("tracker", self.device)
        #     tracker = create_tracker(selftracking_method, reid_weights, device, half)
        #     tracker_list.append(tracker, )
        #     if hasattr(tracker_list[i], 'model'):
        #         if hasattr(tracker_list[i].model, 'warmup'):
        #             tracker_list[i].model.warmup()
        # outputs = [None] * nr_sources
        centers = np.zeros([dataset.nframes,2])
        
        for frame_idx,(path, img, im0s, vid_cap) in enumerate(dataset):
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=self.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=self.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # print("----------------",c)
                        if int(c) > len(names):
                                continue
                            
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    cnt = self.find_center(det[:, :4])
                    centers[frame_idx,:] = cnt
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                            cv2.imwrite(f'{txt_path}.jpg', im0)
                            

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)
        # self.centers = centers

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')
        self.interpolate_empty_values(centers)
        
        self.centers = centers

    def create_new_video(self,frame_size,zoom_factor,file_name):
        full_path = os.path.join("GLIP/",file_name)
        temp_full_path = "GLIP/temp.mp4"
        if os.path.exists(full_path):
            os.remove(full_path)
        if os.path.exists(temp_full_path):
            os.remove(temp_full_path)
        out = cv2.VideoWriter(temp_full_path, cv2.VideoWriter_fourcc(*'XVID'), 60, tuple(frame_size))         
        # dataset = LoadImages(source, img_size=imgsz, stride=model.stride, auto=model.pt)
        # for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        dataset = LoadImages(source, img_size=self.imgsz, stride=1)
        for frame_idx,(path, img, im0s, vid_cap) in enumerate(dataset):
        # for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        # for frame_idx,im0s in enumerate(all_detect_frames):
            if frame_idx%30==0:
                print('writeing frame',frame_idx)
            ccrop_img = self.center_crop(im0s,self.centers[frame_idx],frame_size)
            zoom_im = self.zoom_image(ccrop_img, factor=zoom_factor)
            out.write(zoom_im)

        out.release()
        # cv2.destroyAllWindows()][
        print('Mapping video and audio')
        os.system(f"ffmpeg -i {temp_full_path} -i {source} -c copy -map 0:v:0 -map 1:a:0 -shortest {full_path}")
        # os.remove(temp_full_path)
        print('finished creating video',temp_full_path)
        
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

def main(source, file_name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7x.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.55, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    opt.classes = 32
    # opt.conf-thres = 
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        # if opt.update:  # update all models (to fix SourceChangeWarning)
        #     for opt.weights in ['yolov7.pt']:
        #         detect()
        #         strip_optimizer(opt.weights)
        # else:
        # detect()
        # create_new_video()
        classes = 32
        zoom_factor=1
        # yolo_weights = "yolov5x.pt"
        # device = ""
        # tr = Tracker(classes=classes, device=device)
        # source = "IMG_3853_2.mp4"
        # file_name = 'IMG_3853_2.mp4' # inject name

        # tr.run_tracking_per_frame(source)
        ball_dets = ball_det()
        frame_size = ball_dets.Suported_resolutions['high']
        ball_dets.detect(source)
        new_vid = ball_dets.create_new_video(frame_size,zoom_factor,file_name)
        print("ehere")
        return "GLIP/temp.mp4"

if __name__ == '__main__':
# def main(file_name, source, frame_size):
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', nargs='+', type=str, default='yolov7x.pt', help='model.pt path(s)')
#     parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
#     parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
#     parser.add_argument('--conf-thres', type=float, default=0.55, help='object confidence threshold')
#     parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
#     parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
#     parser.add_argument('--view-img', action='store_true', help='display results')
#     parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
#     parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
#     parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
#     parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
#     parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
#     parser.add_argument('--augment', action='store_true', help='augmented inference')
#     parser.add_argument('--update', action='store_true', help='update all models')
#     parser.add_argument('--project', default='runs/detect', help='save results to project/name')
#     parser.add_argument('--name', default='exp', help='save results to project/name')
#     parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
#     parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
#     opt = parser.parse_args()
#     opt.classes = 32
#     # opt.conf-thres = 
#     print(opt)
#     #check_requirements(exclude=('pycocotools', 'thop'))

#     with torch.no_grad():
#         # if opt.update:  # update all models (to fix SourceChangeWarning)
#         #     for opt.weights in ['yolov7.pt']:
#         #         detect()
#         #         strip_optimizer(opt.weights)
#         # else:
#         # detect()
#         # create_new_video()
#         classes = 32
#         zoom_factor=1
#         # yolo_weights = "yolov5x.pt"
#         device = "cuda:0"#
#         # tr = Tracker(classes=classes, device=device)
#         source = "IMG_3853_2.mp4"
#         file_name = 'IMG_3853_2.mp4' # inject name

#         # tr.run_tracking_per_frame(source)
#         ball_dets = ball_det()
#         frame_size = ball_dets.Suported_resolutions['high']
#         ball_dets.detect(source)
#         new_vid = ball_dets.create_new_video(frame_size,zoom_factor,file_name)
#         print("ehere")
#  ----------------------- for single file ---------------------
    # source = "IMG_3853_2.mp4"
    # file_name = 'IMG_3853_2.mp4' # inject name

    # main(source, file_name)

#  ------------------------ uncomment `for bulk --------------------------------------
        import time
        folder_start_time = time.time()
        for file_name in os.listdir("/mnt/disks/ks3/data/proccessed/H2GvColorado/"): 
            # file_name = 'IMG_3995.mp4'
            if file_name.endswith(".mp4"):
              pass

            else:continue
            source = "/mnt/disks/ks3/data/proccessed/H2GvColorado/"+file_name
            print(source)
            file_start_time = time.time()
            main(source, file_name)
            file_end_time = time.time() - file_start_time
            print("--------------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------------")
            print("---------------------------------",file_end_time,"----------------------------------------------")
            print("--------------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------------")
        print("folder end time", time.time() - folder_start_time)
            
