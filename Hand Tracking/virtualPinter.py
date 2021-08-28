import cv2
import mediapipe as mp
import time
import numpy as np
import HandTrackingModule as htm


cap = cv2.VideoCapture(0)
#3 : WIDTH /4 : HEIGHT
cap.set(3,960)
cap.set(4,540)
smothness=5
pLocX,pLocY=0,0
cLocX,cLocY=0,0
detector=htm.handDetector(maxHands=1, detectionConf=0.80)
tools=cv2.imread('pictures/VIRTUALPINTER.png')
tools = cv2.resize(tools, (960,90) )
drawColor = (0,0,255)
eraserColor = (0,0,0)
brushThickness=10
eraserThickness=30

canvasImg=np.zeros((540,960,3),np.uint8)
register=False
#previous point
xp,yp=0,0

while True:
    success , img = cap.read()
    img=cv2.flip(img,1)
    img=detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) !=0:
        #tip of index finger
        x1,y1=lmList[8][1:]
        #tip of middle finger
        x2,y2=lmList[12][1:]


        fingers =  detector.fingersUp()
        print(fingers)
        if fingers[1] and fingers[2]:

            print("select mode")
            print(x1)
            if y1<90:
                if x1>50 and x1<100:
                    drawColor=(0,0,255)
                if x1>200 and x1<250:
                    drawColor=(0,255,255)
                if x1>370 and x1<425:
                    drawColor=(255,0,0)
                if x1>550 and x1<630:
                    drawColor=eraserColor
                if x1>770 and x1<830:
                    register=True
            cv2.rectangle(img,(x1,y1-20),(x2,y2+20),color=drawColor, thickness=-1)
            xp, yp = 0, 0

        if fingers[1] and fingers[2]==0:

            cv2.circle(img,(x1,y1),15,color=drawColor, thickness=-1)

            print("drawing mode")
            if xp==0 and yp==0:
                xp, yp = x1,y1
            x1 = int(xp + (x1 - xp) / smothness)
            y1 = int(yp + (y1 - yp) / smothness)
            if drawColor==eraserColor:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(canvasImg, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(canvasImg,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp, yp = x1,y1


    img[0:90,0:960]=tools
    img2gray = cv2.cvtColor(canvasImg, cv2.COLOR_BGR2GRAY)
    ret, imgInv = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY_INV)
    imgInv=cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    # Now black-out the area of the drawing
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvasImg)

    #display ******************************

    cv2.imshow("image", img)
    if register:
        reg = cv2.bitwise_or(imgInv, canvasImg)
        cv2.imwrite("pictures/myDrawing.png",reg)
        cv2.destroyWindow('image')
        break

    #cv2.imshow("canva", reg)
    if cv2.waitKey(1)==27:
        break

# Reading an image in default mode
myDrawing = cv2.imread('pictures/myDrawing.png')


# Using cv2.imshow() method
# Displaying the image
cv2.imshow('myDrawing', myDrawing)

# waits for user to press any key
# (this is necessary to avoid Python kernel form crashing)
cv2.waitKey(0)

# closing all open windows
cv2.destroyAllWindows()