import cv2
import mediapipe as mp
import time

import HandTrackingModule as htm

p_time = time.time()
c_time = 0

cap = cv2.VideoCapture(0)
detector=htm.handDetector()
while True:
    success , img = cap.read()
    img=detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) !=0:
        print(lmList[4])
    #frequency stuffs *********************
    c_time=time.time()
    fps=1/(c_time-p_time)
    p_time=c_time
    cv2.putText(img,str(int(fps)),(10,30),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255))
    #display ******************************

    cv2.imshow("image", img)
    cv2.waitKey(1)
