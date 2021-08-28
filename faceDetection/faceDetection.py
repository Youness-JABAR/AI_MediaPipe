import cv2
import mediapipe as mp
import time

#detect our face AND THE POINTS
mpFaceDetection=mp.solutions.face_detection
faceDetection=mpFaceDetection.FaceDetection()


#TO DRAW THE POINTS
mpDraw=mp.solutions.drawing_utils
#mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)  // draw the points and the lines

p_time=0
cap=cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    #img=cv2.resize(img, (800, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)
    print(results)
    if results.detections:
        for id,detection in enumerate(results.detections):
            #mpDraw.draw_detection(img, detection)
            # shape of the frame height width channels
            h, w, c = img.shape
            #bounding box from the class
            bboxC=detection.location_data.relative_bounding_box
            bbox=int(bboxC.xmin * w), int(bboxC.ymin * h) ,\
                    int(bboxC.width * w), int(bboxC.height * h)

            cv2.rectangle(img, bbox,(255, 0, 255),2)
            cv2.putText(img, f'{int(detection.score[0]*100)} %', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
    # frequency stuffs *********************
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, "FPS: "+str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN,  2, (0, 255, 0))
    # display ******************************

    cv2.imshow("image", img)
    #if escape is printed
    if cv2.waitKey(1)==27:
        break
