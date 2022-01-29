from typing import final
import cv2
import time
import numpy as np
fourcc=cv2.VideoWriter_fourcc(*"XVID")
output_file=cv2.VideoWriter("output.avi",fourcc,20.0,(640,480))
cap=cv2.VideoCapture(0)
time.sleep(2)
bg=0
video = cv2.VideoCapture(0) 
image = cv2.imread("me.jpeg") 
for i in range(60):
    ret,bg=cap.read()
bg=np.flip(bg,axis=1)
while (cap.isOpened()):
    ret,img=cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 
    ret, frame = video.read() 
    print(frame)
    frame = cv2.resize(frame, (640, 480)) 
    image = cv2.resize(image, (640, 480)) 
    black_l=np.array([104,153,70])
    black_u=np.array([30,30,0])
    mask = cv2.inRange(frame, black_l, black_u) 
    result=cv2.bitwise_and(frame, frame, mask = mask) 
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
    
    f = frame - result 
    f = np.where(f == 0, image, f) 
    cv2.imshow("video", frame) 
    cv2.imshow("mask", f) 
output_file.release()    
cap.release()    
video.release()
cv2.destroyAllWindows()
