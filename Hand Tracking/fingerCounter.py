import cv2
import mediapipe as mp
import time

import HandTrackingModule as htm

p_time = time.time()
c_time = 0
tipIds=[4,8,12,16,20]
cap = cv2.VideoCapture(0)
detector=htm.handDetector()
while True:
    success , img = cap.read()
    img=cv2.flip(img,1)

    img=detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    fingers=[]
    if len(lmList) !=0:
        #thumb
        if lmList[tipIds[0]][1] < lmList[tipIds[0]-2][1] :
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        print(fingers.count(1))
    cv2.rectangle(img,(0,0),(100,100),color=(237, 196,62 ), thickness=-1)
    cv2.putText(img,str(fingers.count(1)),(40,60),cv2.FONT_HERSHEY_PLAIN,2 ,(0, 0, 0),thickness=2)


    #frequency stuffs *********************
    c_time=time.time()
    fps=1/(c_time-p_time)
    p_time=c_time
    cv2.putText(img,'FPS: '+str(int(fps)),(450,40),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
    #display ******************************

    cv2.imshow("image", img)
    if cv2.waitKey(1)==27:
        break
