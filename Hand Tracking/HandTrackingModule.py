import cv2
import mediapipe as mp
import time
import math

#we have created this module in order to get the position vali=ues of the landmarks easily


class handDetector():
    def __init__(self,mode=False,
               maxHands=2,
               detectionConf=0.5,
               trackConf=0.5):
            self.mode=mode
            self.maxHands=maxHands
            self.detectionConf=detectionConf
            self.trackConf=trackConf
            # import the hand detection
            self.mpHands = mp.solutions.hands
            self.hands = self.mpHands.Hands(self.mode,self.maxHands,self.detectionConf,self.trackConf)
            # to draw the landmarks
            self.mpDraws = mp.solutions.drawing_utils

            #fingersUp
            self.tipIds=[4,8,12,16,20]

    def findHands(self,img,draw=True):

        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(imgRGB)
        #mediapipe.python.solution_base.SolutionOutputs
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw :
                    self.mpDraws.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS )
        return img
    def findPosition(self,img,handNo=0,draw=True):
        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand =self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw :
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList
    def fingersUp(self):
        fingers = []
        # thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #4fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self,img,p1,p2,draw=False,r=15,t=3):
        x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
        x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # hypotenuse
        length = math.hypot(x2 - x1, y2 - y1)
        return length,img,[cx,cy]


def main():
    p_time = time.time()
    c_time = 0

    cap = cv2.VideoCapture(0)
    detector=handDetector()
    while True:
        success , img = cap.read()
        img=detector.findHands(img)
        lmList=detector.findPosition(img)
        if len(lmList) !=0:
            print(lmList[4])
        #frequency stuffs *********************
        c_time=time.time()
        fps=1/(c_time-p_time)
        p_time=c_time
        cv2.putText(img,str(int(fps)),(10,30),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255))
        #display ***************************** *

        cv2.imshow("image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()