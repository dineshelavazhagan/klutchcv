import cv2
import numpy as np
import os
import glob
import argparse
import time
from ctypes import *
import math
import random
import os

import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime
# import pandas as pd
from tool.utilss import *
from tool.darknet2onnx import *

import xml.etree.cElementTree as ET
from natsort import natsorted

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required=True,
#                help="path to input image")
ap.add_argument("-y", "--yolo", required=False,
                help="base path to YOLO directory")  
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
                help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# load the COCO class labels our YOLO model was trained on
# labelsPath = os.path.sep.join([args["yolo"], "obj.names"])
# LABELS = open(labelsPath).read().strip().split("\n")

# # derive the paths to the YOLO weights and model configuration
# weightsPath = os.path.sep.join([args["yolo"], "yolo-obj_last.weights"])
# configPath = os.path.sep.join([args["yolo"], "yolo-obj.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
# net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# # determine only the *output* layer names that we need from YOLO
# ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# session_v3 = onnxruntime.InferenceSession("onnx_saved/yolov3_1_3_416_416_static.onnx")
session_v4_last = onnxruntime.InferenceSession("y4_v2_helmet.onnx")

img_dir="Trichy/wbroad2/"
data_path=os.path.join(img_dir,'*g')
files=glob.glob(data_path)
namesfile = 'y4_helmet.names'
class_names = load_class_names(namesfile)
print("add")
def detect_onn(session, image_src):
    IN_IMAGE_H = session.get_inputs()[0].shape[2]
    IN_IMAGE_W = session.get_inputs()[0].shape[3]

    # Input
    image_src = cv2.imread(image_src)
    print("The model expects input shape: ", session.get_inputs()[0].shape)

    # print(image_src)
    # cv2.imshow("winname", image_src)
    resized = cv2.resize(image_src, (IN_IMAGE_W, IN_IMAGE_H),
                         interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    # print("Shape of the network input: ", img_in.shape)

    # Compute
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: img_in})

    boxes = post_processing(img_in, 0.4, 0.6, outputs)
    return boxes

print(files)
def get_object_params(bb, size):
        image_width = 1.0 * size[0]
        image_height = 1.0 * size[1]
        # box = obj.box
        absolute_x = box[0] + 0.5 * (box[2] - box[0])
        absolute_y = box[1] + 0.5 * (box[3] - box[1])

        absolute_width = box[2] - box[0]
        absolute_height = box[3] - box[1]

        x = absolute_x / image_width
        y = absolute_y / image_height
        width = absolute_width / image_width
        height = absolute_height / image_height

        return [x, y, width, height]

files = natsorted(files)

for f1 in files:
    # print(f1)
    # load our input image and grab its spatial dimensions
    try:
      image=cv2.imread(f1)
      # print(image.shape)

      # construct a blob from the input image and then perform a forward
      # pass of the YOLO object detector, giving us our bounding boxes and
      # associated probabilities
      # blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
      #                              swapRB=True, crop=False)
      # net.setInput(blob)
      # layerOutputs = net.forward(ln)

      # loop over each of the layer outputs
      # for output in layerOutputs:
          # loop over each of the detections
      output = detect_onn(session_v4_last, f1)
      # print(output)
      if len(output[0]) <1:
        continue
      img_name =  f1.split("/")[-1]
      img_name = img_name.split(".")[0]
      root = ET.Element("annotation")
      folder = ET.SubElement(root, "folder").text = f1.split("/")[-2]
      filename = ET.SubElement(root, "filename").text = f1.split("/")[-1]
      path = ET.SubElement(root, "path").text = f1
      source = ET.SubElement(root, "source")
      ET.SubElement(source, "database").text="Unknown"
      size = ET.SubElement(root, "size")
      ET.SubElement(size, "width").text = str(image.shape[1])
      ET.SubElement(size, "height").text = str(image.shape[0])
      ET.SubElement(size, "depth").text = "3"
      ET.SubElement(root, "segmented").text = "0"
      for detection in output[0]:  
        # extract the class ID and confidence (i.e., probability) of
        # the current object detection
        # print(detection)
        # scores = detection[5:]
        classID = detection[-1]
        confidence = detection[-2]
        box = detection[0:4]
        width = image.shape[1]
        height = image.shape[0]
        # box[0] = abs(box[0] - box[2]/2)
        # box[1] = abs(box[1] - box[3]/2)
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        # bb=[x1,y1,x2,y2]
        # box = get_object_params(bb, image.shape)
        # get upper left corner
        

        # filter out weak predictions by ensuring the detected
        # probability is greater than the minimum probability
        if confidence > args["confidence"]:
            # write output files
            class_dir ='/workspace/for_test/'

            if not os.path.exists(class_dir):
                os.makedirs(class_dir)
            path = os.path.join(class_dir, f1.split('/')[-1][:-4])
            # print("writing")
            # cv2.imwrite(path + '.jpg', image)
            print(path)
            import xml.etree.cElementTree as ET

            
            object = ET.SubElement(root, "object")
            ET.SubElement(object, "name").text= class_names[classID]
            ET.SubElement(object, "pose").text="Unspecified"
            ET.SubElement(object, "truncated").text="0"
            ET.SubElement(object, "difficult").text="0"
            bndbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bndbox, "xmin").text= str(x1)
            ET.SubElement(bndbox, "ymin").text= str(y1)
            ET.SubElement(bndbox, "xmax").text= str(x2)
            ET.SubElement(bndbox, "ymax").text= str(y2)
      tree = ET.ElementTree(root)
      tree.write(f1.split(".")[0]+".xml",encoding="utf-8", xml_declaration=True)
    except:
      os.remove(f1)
      pass