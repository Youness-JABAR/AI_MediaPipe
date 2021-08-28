import cv2
import mediapipe as mp
import time




class FaceMeshDetector():
    def __init__(self,staticMode=False,
               maxFaces=1,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):


        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode,self.maxFaces,self.min_detection_confidence,self.min_tracking_confidence)
        self.mpDraws = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraws.DrawingSpec(thickness=1, circle_radius=2)
    def findFaceMesh(self,img,draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        # multiple Faces
        lmListFaces = []
        if self.results.multi_face_landmarks:

            for faceLms in self.results.multi_face_landmarks:
                # one face 468 POINT
                lmList=[]
                if draw:
                    self.mpDraws.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)
                for id, lm in enumerate(faceLms.landmark):
                    # shape of the frame height width channels
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    lmList.append([cx,cy])
                    cv2.putText(img,str(id), (cx, cy),cv2.FONT_HERSHEY_PLAIN,0.4, (255, 0, 255),1)
                    #if (id == 0):
                    #    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                lmListFaces.append(lmList)

        return img,lmListFaces

def main():
    cap = cv2.VideoCapture(0)
    # previous and current time to calculate the frames per second(frame rate):
    p_time = time.time()
    c_time = 0
    detector=FaceMeshDetector()
    while True:
        success, img = cap.read()
        img,lmListFaces=detector.findFaceMesh(img,False)
        print(len(lmListFaces))
        # frequency stuffs *********************
        c_time = time.time()
        fps = 1 / (c_time - p_time)
        print(c_time - p_time)
        p_time = c_time
        cv2.putText(img, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255))
        # display ******************************

        cv2.imshow("image", img)
        if cv2.waitKey(1) == 27:
            break


if __name__=="__main__":
    main()