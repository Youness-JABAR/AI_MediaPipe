import autopy as autopy
import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm

wCam,hCam=640,480
cap = cv2.VideoCapture(0)
smothness=5
pLocX,pLocY=0,0
cLocX,cLocY=0,0
#3 : WIDTH /4 : HEIGHT
cap.set(3,wCam)
cap.set(4,hCam)
detector=htm.handDetector(maxHands=1, detectionConf=0.80)
wScr,hScr=autopy.screen.size()
print(wScr,hScr)
frameR=100 #reduction

while True:

    success , img = cap.read()
    img=cv2.flip(img,1)
    img=detector.findHands(img)
    cv2.rectangle(img, (frameR,frameR), (wCam - frameR, hCam - frameR), color=(255,0,255))

    lmList = detector.findPosition(img, draw=False)
    if len(lmList) !=0:
        #tip of index finger
        x1,y1=lmList[8][1:]
        #tip of middle finger
        x2,y2=lmList[12][1:]

        fingers =  detector.fingersUp()
        print(fingers)

        if fingers[1] and fingers[2]==0:
            x3=np.interp(x1,(frameR,wCam - frameR),(0,wScr))
            y3=np.interp(y1,(frameR,hCam - frameR),(0,hScr))
            cLocX=pLocX+(x3-pLocX)/smothness
            cLocY=pLocY+(y3-pLocY)/smothness
            autopy.mouse.move(cLocX,cLocY)
            cv2.circle(img,(x1,y1),15,color=(0,0,255), thickness=-1)
            pLocX, pLocY =cLocX, cLocY

            print("moving mode")

        if fingers[1] and fingers[2]:
            print("clicking mode")
            length,img,center =detector.findDistance(img,8,12,True)
            print(length)
            if length<40:
                cv2.circle(img, (center[0], center[1]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()




    cv2.imshow("image", img)
    if cv2.waitKey(1)==27:
        break