"""
==========================================
Program to recognize face(squid game) using OpenCV

Creators:
Tomasz Samól (Plastikowy)
Sebastian Lewandowski (SxLewandowski)
==========================================
Prerequisites:
Before you run program, you need to install Numpy and opencv-python packages.
You can use for example use PIP package manager do to that:
pip install numpy
pip install opencv-python
==========================================
"""

import cv2
import numpy as np
import mediapipe as mp

# TODO:
# capture video
# recognize face
# recognize hand
# outline face (red)
# outline hand (green)
# print crosshair on face
# print aiming at moving target

# declare video capturing variable
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# declare variable for mediapipe hands modules
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mpDrawStyles = mp.solutions.drawing_styles

# def captureVideo():


while (cap.isOpened()):
    # captureVideo()
    ret, frame = cap.read()
    cv2.startWindowThread()
    cv2.namedWindow('Camera Capture')
    cv2.imshow('Camera Capture', frame)

    # convert colours captured from camera to RGB
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:
            print(results.multi_hand_landmarks[0])
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS,
                                  mpDrawStyles.get_default_hand_landmarks_style(),
                                  mpDrawStyles.get_default_hand_connections_style())

    cv2.imshow('Camera Capture', frame)
    readKey = cv2.waitKey(1)
    if readKey == ord('b'):
        cv2.imwrite('screenshot_now.jpg', frame)
        print('Zrzut zrobiony')
    elif readKey == ord('t'):
        cv2.circle(frame, (320,240), 30, (255,0,128), 5)
        cv2.imwrite('screenshot_now.jpg', frame)
        print('Rysuj kropke')

cap.release()
cv2.destroyAllWindows()
