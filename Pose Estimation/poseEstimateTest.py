import cv2

import time
import PoseEstimationModule as pm


p_time = 0
cap = cv2.VideoCapture("PoseVid/2.mp4")
detector=pm.poseDetector()
while True:
    success, img = cap.read()
    img = cv2.resize(img, (800, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    img = detector.findPose(img)
    lmList=detector.findPosition(img,draw=False)
    if len(lmList) != 0:
        print(lmList[14])
        cv2.circle(img, (lmList[14][1],lmList[14][2]), 5, (255, 0, 255), cv2.FILLED)

    # frequency stuffs *********************
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255))
    # display ******************************

    cv2.imshow("image", img)
    #if escape is printed
    if cv2.waitKey(1)==27:
        break
