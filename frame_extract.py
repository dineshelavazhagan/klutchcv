import cv2
import os

import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
path = "../data/raw/all_vid/"

for vid_path in os.listdir(path):
    # if "" not in vid_path
    # print(vid_path)
    vs = cv2.VideoCapture(path+vid_path)
    count = 1
    print(path+vid_path)
    while True:
        try:
        
            _, frame = vs.read()
            if not _:
                break
            count += 1
            # print(count)
            # cv2.imshow("fra,e", frame)
            # cv2.waitKey(0)
            name = vid_path.split(".")[0]
            print("frame_ext/"+name+"/"+str(count)+".png")
            cv2.imwrite("frame_ext/"+name+"_"+str(count)+".png", frame)
        except:
            break
    winsound.Beep(freq, duration)
    

