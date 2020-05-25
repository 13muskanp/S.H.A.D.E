# python3 fg.py --prototxt deploy.prototxt --net res10_300x300_ssd_iter_140000.caffemodel --model mobilenet_thin --resize 432x368 --camera 0
# python3 run_webcam.py --model=mobilenet_thin --resize=432x368 --camera=0

from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import numpy as np
import argparse
import imutils
import time
import cv2

import csv
import time
start_time = time.time()

def sorted_set(array, array2, assume_unique=False):
    ans = np.setdiff1d(array, array2, assume_unique).tolist()
    if assume_unique:
        return sorted(ans)
    return ans


fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default='deploy.prototxt',
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-n", "--net",   default='res10_300x300_ssd_iter_140000.caffemodel', 
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")

ap.add_argument("--camera", type=int, default=0)

ap.add_argument("--resize", type=str, default='432x368',
                    help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
ap.add_argument("--resize-out-ratio", type=float, default=4.0,
                    help='if provided, resize heatmaps before they are post-processed. default=1.0')

ap.add_argument("--model", type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
ap.add_argument("--show-process", type=bool, default=False,
                    help='for debug purpose, if enabled, speed for inference is dropped.')

ap.add_argument("--tensorrt", type=str, default="False",
                    help='for tensorrt process.')

args = vars(ap.parse_args())


ct = CentroidTracker()
(H, W) = (None, None)

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["net"])



w, h = model_wh(args["resize"])
if w > 0 and h > 0:
    e = TfPoseEstimator(get_graph_path(args["model"]), target_size=(w, h), trt_bool=str2bool(args["tensorrt"]))
else:
    e = TfPoseEstimator(get_graph_path(args["model"]), target_size=(432, 368), trt_bool=str2bool(args["tensorrt"]))

###########################################
import cv2
import numpy as np
import glob
from scipy.spatial import distance
from imutils import face_utils
from keras.models import load_model
import tensorflow as tf

from fr_utils import *
from inception_blocks_v2 import *

#with CustomObjectScope({'tf': tf}):
FR_model = load_model('nn4.small2.v1.h5')
print("Total Params:", FR_model.count_params())

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

threshold = 0.25

face_database = {}

for name in os.listdir('images'):
    for image in os.listdir(os.path.join('images',name)):
        identity = os.path.splitext(os.path.basename(image))[0]
        face_database[identity] = fr_utils.img_path_to_encoding(os.path.join('images',name,image), FR_model)

print(face_database)
###########################################

cam = cv2.VideoCapture(args["camera"])
Detections_prev = []
while True:
    ret_val, image = cam.read()
    Detections = []
    # image = cv2.flip(image, 1)  #################
    faces = face_cascade.detectMultiScale(cv2.flip(image, 1), 1.3, 5) ################
    
    image = imutils.resize(image, width=400)
    
    if W is None or H is None:
        (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []

    for i in range(0, detections.shape[2]):
        if detections[0, 0, i, 2] > args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))

            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 255, 0), 2)

    objects = ct.update(rects)
####################################################################
    for(x,y,w,h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)
        roi = image[y:y+h, x:x+w]
        encoding = img_to_encoding(roi, FR_model)
        min_dist = 100
        identity = None

        

####################################################################
### Shifting these by 1 tab
        for (objectID, centroid) in objects.items():
            text = "Muskan ID {}".format(objectID)
        Detections.append(text)
            #print(Detections)
        ######
        for(name, encoded_image_name) in face_database.items():
            dist = np.linalg.norm(encoding - encoded_image_name)
            if(dist < min_dist):
                min_dist = dist
                identity = name
            print('Min dist: ',min_dist)

        if min_dist < 0.1:
            cv2.putText(image, "Face : " + identity[:-1], (x, y - 50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            cv2.putText(image, "Dist : " + str(min_dist), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        else:
            cv2.putText(image, 'Muskan', (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2)
        ######
        diff = [i for i in Detections_prev + Detections if i not in Detections or i not in Detections_prev]#list(set(Detections) - set(Detections_prev))
        print(diff)
        if (len(diff)> 0):
            with open('record.csv', 'a', newline='') as updatecsv:
                writer_n = csv.writer(updatecsv, delimiter=",", lineterminator='\n')
                Row = [text, time.time() - start_time]
                writer_n.writerows([Row])#map(lambda x, y: [], Row
        """
        with open('record.csv', 'w', newline='') as outcsv:
            reader = csv.reader(outcsv)
            Row = [text, time.time() - start_time]
            writer.writerows(Row)   
        """ 
        Detections_prev = Detections

        cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args["resize_out_ratio"])

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
########## shift end
    cv2.imshow("image", image)  
    fps_time = time.time()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
