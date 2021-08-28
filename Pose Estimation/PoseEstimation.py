import cv2
import mediapipe as mp
import time

#detect our pose AND THE POINTS
mpPose=mp.solutions.pose
pose=mpPose.Pose()
#    results=pose.process(imgRGB)
#    results.pose_landmarks   //return the landmark

#TO DRAW THE POINTS
mpDraw=mp.solutions.drawing_utils
#mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)  // draw the points and the lines

p_time=0
cap=cv2.VideoCapture("PoseVid/2.mp4")

while True:
    success, img = cap.read()
    img=cv2.resize(img, (800, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(imgRGB)
    print(results)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            # shape of the frame height width channels
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            print(id, cx, cy)
            if (id == 0):
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

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
