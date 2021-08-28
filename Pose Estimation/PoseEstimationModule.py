import cv2
import mediapipe as mp
import time
import  math
# class to detect the pose and find all the landmarks
class poseDetector():
    def __init__(self,mode=False,
               upBody=False,
               smooth=True,
               detectionConf=0.5,
               trackConf=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionConf = detectionConf
        self.trackConf = trackConf
        # import the pose detection
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.upBody, self.smooth, self.detectionConf, self.trackConf)
        # to draw the landmarks
        self.mpDraw = mp.solutions.drawing_utils
    def findPose(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img
    def findPosition(self,img,draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lmList
    def findAngle(self,img,p1,p2,p3,draw=True):
        x1,y1=self.lmList[p1][1:]
        x2,y2=self.lmList[p2][1:]
        x3,y3=self.lmList[p3][1:]
        angle=-math.degrees( math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
        if angle<0:
            angle+=360
        cv2.putText(img,f'{int(angle)}',(x2+10,y2),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
        #print(angle)

        if draw:
            cv2.circle(img,(x1,y1),10,(0,0,255), cv2.FILLED)
            cv2.circle(img,(x1,y1),15,(0,0,255), 2)
            cv2.circle(img,(x2,y2),10,(0,0,255), cv2.FILLED)
            cv2.circle(img,(x2,y2),15,(0,0,255), 2)
            cv2.circle(img,(x3,y3),10,(0,0,255), cv2.FILLED)
            cv2.circle(img,(x3,y3),15,(0,0,255), 2)
            cv2.line(img,(x1,y1),(x2,y2),(0,0,0),2)
            cv2.line(img,(x3,y3),(x2,y2),(0,0,0),2)
        return angle

def main():
    p_time = 0
    cap = cv2.VideoCapture("PoseVid/2.mp4")
    detector=poseDetector()
    while True:
        success, img = cap.read()
        img = cv2.resize(img, (800, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        img = detector.findPose(img)
        lmList=detector.findPosition(img,draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 5, (255, 0, 255), cv2.FILLED)
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


#if we run this code it will execute the __main__ part else will not
if __name__=="__main__":
    main()