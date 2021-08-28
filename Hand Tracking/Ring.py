import cv2
import mediapipe as mp
import time
import math
import numpy as np
import HandTrackingModule as htm

detector=htm.handDetector()




ring = cv2.imread('pictures/ringtransparent.PNG')
#h_ring, w_ring, _ = ring.shape
#ring = cv2.resize(ring, (int(h_ring / 2), int(w_ring / 2)))
h_ring, w_ring, _ = ring.shape
rotation = cv2.getRotationMatrix2D((w_ring // 2, h_ring // 2), 90, 1.0)
ring = cv2.warpAffine(ring, rotation, (w_ring, h_ring))

#img = cv2.imread('pictures/myHand.png')

cap = cv2.VideoCapture(0)
while True:
    success , img = cap.read()


    img=detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) !=0:
        print(lmList[14])
        x1, y1 = lmList[14][1], lmList[14][2]
        x2, y2 = lmList[13][1], lmList[13][2]

        #Adjust the ring scale
        ring = cv2.imread('pictures/ringtransparent.PNG')
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        ring = cv2.resize(ring, (int(dist / 2), int(dist / 2)))
        h_ring, w_ring, _ = ring.shape

        #dterminate the angle of the finger
        angle = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
        print(angle)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img,(cx,cy),5,(0,255,0),-1)
        top_y=cy-h_ring//2
        buttom_y = top_y + h_ring
        left_x = cx - w_ring // 2
        right_x = left_x + w_ring
        #cv2.rectangle(img, (left_x, top_y), (right_x, buttom_y), color=(255, 0, 0), thickness=3)

        #rotate the ring and put it on the frame
        rotation = cv2.getRotationMatrix2D((w_ring // 2, h_ring // 2), -angle+90, 1.0)
        ringRot = cv2.warpAffine(ring, rotation, (w_ring, h_ring))
        frame = img[top_y:buttom_y, left_x:right_x]
        #frameRot = cv2.warpAffine(frame, rotation, (w_ring, h_ring))
        #cv2.imshow("frame", frameRot)
        #cv2.imshow("ringROT", ringRot)
        result = cv2.addWeighted(frame, 1, ringRot, 0.9, 0)

        img[top_y:buttom_y,left_x:right_x] = result

        #cv2.imshow("res", result)


    #cv2.imshow('ring',ring)
    cv2.imshow('img',img)
    cv2.waitKey(1)
