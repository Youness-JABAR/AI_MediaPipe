import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# import the hand detection
mpHands=mp.solutions.hands
hands=mpHands.Hands()
#to draw the landmarks
mpDraws=mp.solutions.drawing_utils
#previous and current time to calculate the frequence per second
p_time = time.time()
c_time = 0



while True:
    success, img = cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    #mediapipe.python.solution_base.SolutionOutputs
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                print(lm)
                #shape of the frame height width channels
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                #print(id,cx,cy)
                if(id==0):
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

            mpDraws.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS )

    #frequency stuffs *********************
    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img,str(int(fps)),(10,30),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255))
    #display ******************************

    cv2.imshow("image", img)
    cv2.waitKey(100)

