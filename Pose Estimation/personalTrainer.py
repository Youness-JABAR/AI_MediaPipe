import cv2
import time
import math
import numpy as np
import PoseEstimationModule as pm


detector=pm.poseDetector()

cap=cv2.VideoCapture('PoseVid/HandTraining.mp4')
#0 : down and 1 : up
direction=0
count=0
while True:
    success,img=cap.read()
    img=cv2.resize(img,(400,500))
    #img=cv2.flip(img,1)
    img=detector.findPose(img,draw=False)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        #detector.findAngle(img,12,14,16)
        angle=detector.findAngle(img,11,13,15)
        per=np.interp(angle,(75,140),(100,0))
        if per==100 and direction==1:
            count+=0.5
            direction=0
        if per==0 and direction==0:
            count+=0.5
            direction=1

        print(int(count))
        cv2.rectangle(img, (0, 0), (40, 40), color=(237, 196, 62), thickness=-1)
        cv2.putText(img,f'{int(count)}',(10,30),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)



    cv2.imshow("video",img)
    if cv2.waitKey(1)==27:
        break

