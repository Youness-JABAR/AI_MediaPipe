'''
ret,frame = cap.read() # return a single frame in variable `frame`

while(True):
    ret, frame = cap.read()  # return a single frame in variable `frame`
    cv2.imshow('img1',frame) #display the captured image
    if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
        img=frame
        cv2.destroyAllWindows()
        break
'''