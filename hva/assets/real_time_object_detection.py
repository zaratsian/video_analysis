
##########################################################################################################################
#
#   Real-time Video Object Detection
#
#   USAGE
#   python real_time_object_detection.py
#
##########################################################################################################################

import os
from imutils.video import VideoStream
#from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
import time

##########################################################################################################################
#
#    External Dependencies:
#
#    docker cp ~/Downloads/IMG_7942.MOV hva:/assets/.
#    docker cp ~/Downloads/MobileNetSSD_deploy.caffemodel hva:/assets/.
#    docker cp ~/Downloads/MobileNetSSD_deploy.prototxt.txt hva:/assets/.
#
##########################################################################################################################

print(os.listdir('/assets'))    

##########################################################################################################################

# Initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", 
    "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the trained and serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('/assets/MobileNetSSD_deploy.prototxt.txt', '/assets/MobileNetSSD_deploy.caffemodel')

# This is used for testing when running video stream from webcam 
# Initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
#print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
#time.sleep(2.0)
#fps = FPS().start()

print("Load video file")
vs = cv2.VideoCapture('/assets/IMG_7942.MOV')

# Loop through each frame of the video stream
while True:
    # Grab the frame from the threaded video stream and resize it to X pixels
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=600)
    
    # Convert frame dimensions to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
    
    # Pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()
    
    # Loop over the detections
    found_object = False
    for i in np.arange(0, detections.shape[2]):
        
        # Get the prediction confidence (probability) level
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.2:
            # Extract the index of the class label from the
            # detections, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            if CLASSES[idx] in ['horse','car']:
                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx],
                    confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
            if CLASSES[idx]=='car':
                found_object = True
    
    # Show the output frame
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if found_object:
        time.sleep(2)
    
    # Break loop when 'q' button is pressed
    if key == ord("q"):
        break
    
    # update the FPS counter
    #fps.update()

# stop the timer and display FPS information
#fps.stop()
#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[ INFO ] Processing Complete")

# do a bit of cleanup
cv2.destroyAllWindows()
try:
    vs.release()
    #vs.stop()
except:
    pass

#ZEND