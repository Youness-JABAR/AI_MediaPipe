import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# import the hand detection
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
#to draw the landmarks
mpDraws=mp.solutions.drawing_utils
drawSpec=mpDraws.DrawingSpec(thickness=1,circle_radius=2)
#previous and current time to calculate the frames per second(frame rate):
p_time = time.time()
c_time = 0



while True:
    success, img = cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=faceMesh.process(imgRGB)
    #mediapipe.python.solution_base.SolutionOutputs
    #print(results.multi_hand_landmarks)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            for id,lm in enumerate(faceLms.landmark):
                #shape of the frame height width channels
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)
                if(id==0):
                    cv2.circle(img,(cx,cy),3,(255,0,255),cv2.FILLED)

            mpDraws.draw_landmarks(img,faceLms,mpFaceMesh.FACE_CONNECTIONS,drawSpec,drawSpec )

    #frequency stuffs *********************
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    print(c_time - p_time)
    p_time = c_time
    cv2.putText(img, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255))
    #display ******************************

    cv2.imshow("image", img)
    if cv2.waitKey(1)==27:
        break

