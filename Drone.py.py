from __future__ import with_statement
from __future__ import division
from __future__ import absolute_import
from os import sys, path
from io import open
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

#-- Dependencies for video processing
import time
import math
import argparse
import cv2
import numpy as np
from imutils.video import FPS

#-- Dependencies for commanding the drone
from dronekit import connect, VehicleMode, LocationGlobalRelative, Command, LocationGlobal
from pymavlink import mavutil

######-----------------------Dronekit-----------------------######
#-- Parse argument to connect the drone via terminal window
parser = argparse.ArgumentParser()
parser.add_argument(u'--connect', default = u'')
args = parser.parse_args()

#-- Connect to the drone
print u'Connecting...'
vehicle = connect(args.connect, wait_ready = True)  

#-- Function to controll velocity of the drone
def send_local_ned_velocity(x, y, z):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, # frame
        __builtins__.long("0000111111000111", 2), 
        0, 0, 0, 
        x, y, z, # x, y, z velocity in m/s
        0, 0, 0,
        0, 0)    

    # send command to the drone
    vehicle.send_mavlink(msg)
    vehicle.flush()


######-----------------------OpenCV-----------------------######
cap = cv2.VideoCapture(0) # capture video from web camera
#-- Start saving output video
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(u"output_pred.mp4", fourcc, 15.0,(640,480))

######-----------------------Object detecting and tracking-----------------------######
#-- Parameters for YOLO model
whT = 320 #width and height for target
confidenceThreshold = 0.8
nmsThreshold =0.3
boxTracker=[]
index = 0
confidence = 1.5

#-- Initialize COCO datasets for detecting only a person
classesFile = u'cocoperson.names'
classNames = []
with open(classesFile,u'rt') as f:
    classNames =f.read().rstrip(u'\n').split(u'\n')

#-- Import config and weights file for detecting algorithm
modelConfiguration =u'yolov3.cfg'
modelWeights = u'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #GPU will perform the processing
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#-- Detect object between the number of frames that you want to skip
totalFrames = 0 # to count frames
skip = 30 # skip frames which means after every 30 frames, detecting algorithm will detect a person

#-- Initialize MOSSE tracking algorithm
tracker = cv2.TrackerMOSSE_create()

#-- Draw bounding box for visualization
def drawBox(img, box,index,confidence):
    x,y,w,h = int(box[0]), int(box[1]),int(box[2]),int(box[3])
    cv2.rectangle(img, (x,y), ((x+w),(y+h)), (255,0,255), 1, cv2.LINE_AA)
    cv2.putText(img,"{classNames} {confidence}%".format(classNames = classNames[index].upper(), confidence = int(confidence*100)), (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),1)
    cv2.circle(img, (int((x+w/2)),int((y+h/2))), 4, (255, 100, 255), -1)
    cv2.putText(img,u"Status: Tracking",(75,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
    cv2.circle(img, (int(640/2),int(480/2)), 4, (255, 100, 255), -1)
    cv2.line(img,(int(640/2),int(480/2)), (int((x+w/2)),int((y+h/2))), (255,255,255), 1)

fps = FPS().start() # Frames Per Second

######-----------------------Main function-----------------------######
while True:
    timer=cv2.getTickCount()
    success, img=cap.read()

    #object detection method to aid our tracker
    if totalFrames % skip == 0 or (boxTracker == False):
        cv2.putText(img,u"Status: Detecting",(75,75),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
        trackers = []
        blob = cv2.dnn.blobFromImage(img, 1/255,(whT,whT),[0,0,0], 1,crop=False)
        net.setInput(blob)
        layerNames = net.getLayerNames() # get all the names of network layers and we extract only names of output layers
        outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()] #get layer names
        outputs = net.forward(outputNames)

        #find objects
        hT, wT, cT = img.shape
        print u"hT",hT
        print wT
        print cT
        boxes = []
        classIds = []
        confidences = []
        distance_A =0

        person = img

        for output in outputs:
            for detection in output:
                scores = detection[5:6] # to detect only person
                classId= np.argmax(scores)
                confidence = scores[classId]
                if confidence > confidenceThreshold:
                    w,h = int(detection[2]*wT), int(detection[3]*hT) # the pixcel values
                    x,y = int(detection[0]*wT - w/2), int(detection[1]*hT - h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidences.append(float(confidence))
        #print(len(boxes))
        indices = cv2.dnn.NMSBoxes(boxes,confidences,confidenceThreshold,nmsThreshold)
        print confidences
        print type(confidences)
        if confidence: 
            print max(confidences)
            print confidences.index(max(confidences))

        for i in indices:
            i = i[0] #there's an extra bracket in i so we have to remove that
            box = boxes[i]
            x,y,w,h=  box[0],box[1],box[2],box[3]
            x1=int(x-w * 0.5) # Start X coordinate
            y1=int(y-h * 0.5)# Start Y coordinate
            x2=int(x+w * 0.5)# End X coordinate
            y2=int(y+h * 0.5)# End y 
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,9,255),2)
            cv2.putText(img,"{classNames} {confidence}%".format(classNames=classNames[classIds[i]].upper(), confidence= int(confidences[i]*100)), (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

            ######-----------------------Estimated distance----------------------######
            distance = 474*(170/h)

            boxTracker= box
            index = classIds[i]
            confidence = confidences[i]
            person = img[y:y+h,x:x+w]
        if person.any(): 
            cv2.imshow(u"extract", person)
        if boxTracker:
            del(tracker)
            tracker = cv2.TrackerMOSSE_create()

    else:# loop over the trackers
            if boxTracker:
                tracker.init(img, tuple(boxTracker))
            #print(tracker)
            succes, boundingBox =tracker.update(img)
            print boundingBox
            drawBox(img,boundingBox,index,confidence)
            cv2.putText(img,"Estimated distance {distance}cm".format(distance=int(distance)), (75,95),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255),2)

            ######-----------------------Controlling drone-----------------------######
            #-- Fly forward and backward
            if distance > 600:
                send_local_ned_velocity(1,0,0) # x is positive number so the drone flying forward 1m/s in 1 second duration
            if distance <300:
                send_local_ned_velocity(-1,0,0) # x is negative number so the drone flying backward 1m/s in 1 second duration
            #-- Fly right and left
            if int(boundingBox[0]+boundingBox[2]/2) > 360: # person is in the right so it needs to fly right
                send_local_ned_velocity(0,1,0) # y is positive number so the drone flying right 1m/s in 1 second duration
            if int(boundingBox[0]+boundingBox[2]/2) < 280: # person is in the left so it needs to fly left
                send_local_ned_velocity(0,-1,0) # y is negative number so the drone flying left 1m/s in 1 second duration

    totalFrames+=1

    #fps
    fps.update()

    fps1 = cv2.getTickFrequency()/(cv2.getTickCount()-timer)
    cv2.putText(img,unicode(int(fps1)),(75,50),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    out.write(img)

    cv2.imshow(u'Image',img)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord(u"q"):
        break
fps.stop()
print u"[INFO] elapsed time: {:.2f}".format(fps.elapsed())
print u"[INFO] approx. FPS: {:.2f}".format(fps.fps())

cap.release()
out.release()
