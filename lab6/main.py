import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

#TODO:
#capture video
#recognize face
#recognize hand
#outline face (red)
#outline hand (green)
#print crosshair on face
#print aiming at moving target

while(cap.isOpened()):
    ret, frame1 = cap.read()
    #print('hello')
    cv2.imshow('Kamerka', frame1)

    readKey = cv2.waitKey(10)


cap.release()
cv2.destroyAllWindows()