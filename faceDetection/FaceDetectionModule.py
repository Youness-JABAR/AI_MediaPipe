import cv2
import mediapipe as mp
import time

#we have created this module in order to get the position vali=ues of the landmarks easily


class FaceDetector():
    def __init__(self,minDetectioncon=0.5):
            self.minDetectioncon=minDetectioncon

            # import the hand detection
            self.mpFaceDetection=mp.solutions.face_detection
            self.faceDetection=self.mpFaceDetection.FaceDetection(minDetectioncon)

            # to draw the landmarks
            self.mpDraws = mp.solutions.drawing_utils
    def findFaces(self,img,draw=True):

        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.faceDetection.process(imgRGB)
        bboxs=[]
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                # mpDraw.draw_detection(img, detection)
                # shape of the frame height width channels
                h, w, c = img.shape
                # bounding box from the class
                bboxC = detection.location_data.relative_bounding_box
                bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                bboxs.append([id,bbox,detection.score])
                if draw:
                    img=self.fancyDraw(img,bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)} %', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                            2, (0, 255, 0))


        return img,bboxs
    #length,thickness
    def fancyDraw(self,img,bbox,l=30,t=5):
        x,y,w,h=bbox
        x1,y1=x+w,y+h
        cv2.rectangle(img, bbox, (255, 0, 255), 1)
        #top left
        cv2.line(img,(x,y),(x+l,y),(255,0,255),t)
        cv2.line(img,(x,y),(x,y+l),(255,0,255),t)
        #bottom right
        cv2.line(img,(x1,y1),(x1-l,y1),(255,0,255),t)
        cv2.line(img,(x1,y1),(x1,y1-l),(255,0,255),t)
        #bottom left
        cv2.line(img,(x,y1),(x+l,y1),(255,0,255),t)
        cv2.line(img,(x,y1),(x,y1-l),(255,0,255),t)
        #top right
        cv2.line(img,(x1,y),(x1-l,y),(255,0,255),t)
        cv2.line(img,(x1,y),(x1,y+l),(255,0,255),t)
        return img

def main():
    p_time = time.time()
    c_time = 0

    cap = cv2.VideoCapture(0)
    detector=FaceDetector()
    while True:
        success , img = cap.read()
        img,bboxs=detector.findFaces(img)

        #frequency stuffs   frame rate *********************
        c_time=time.time()
        fps=1/(c_time-p_time)
        p_time=c_time
        cv2.putText(img, "FPS: " + str(int(fps)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0,255))
        #display ***************************** *

        cv2.imshow("image", img)
        # if escape is printed
        if cv2.waitKey(1) == 27:
            break


if __name__ == "__main__":
    main()