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

import cv2 as cv
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
cap = cv.VideoCapture(0, cv.CAP_DSHOW)
# declare variable for mediapipe hands modules
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
mpDrawStyles = mp.solutions.drawing_styles

cascade_path = 'haar_cascade.xml'
face_cascade = cv.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise IOError('Unable to load the cascade classifier xml file')

# def captureVideo():


while cap.isOpened():
    # captureVideo()
    ret, frame = cap.read()
    _, frame_2 = cap.read()
    cv.startWindowThread()
    cv.namedWindow('Camera Capture')
    cv.imshow('Camera Capture', frame)

    # convert colours captured from camera to RGB
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # find difference between two frames
    diff = cv.absdiff(frame, frame_2)
    # convert the frame to grayscale
    diff_gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    # apply some blur to smoothen the frame
    diff_blur = cv.GaussianBlur(diff_gray, (5, 5), 0)
    # get the binary image
    _, thresh_bin = cv.threshold(diff_blur, 20, 255, cv.THRESH_BINARY)
    # find contours
    contours, hierarchy = cv.findContours(thresh_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # draw the bounding box when the motion is detected
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        if cv.contourArea(contour) > 300:
            # cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # draw crosshair when motion is detected
            faces_detect = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)
            for (x, y, w, h) in faces_detect:
                # cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness=4)
                cv.circle(frame, (int(x + w / 2), int(y + h / 5)), int(h / 6), (0, 0, 255), thickness=4)
                cv.line(frame, (int(x + w / 3), int(y + h / 5)), (int(x + 2 * w / 3), int(y + h / 5)), (0, 0, 255),
                        thickness=4)
                cv.line(frame, (int(x + w / 2), int(y + h / 2.8)), (int(x + w / 2), int(y + h / 20)), (0, 0, 255),
                        thickness=4)

    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:
            print(results.multi_hand_landmarks[0])
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS,
                                  mpDrawStyles.get_default_hand_landmarks_style(),
                                  mpDrawStyles.get_default_hand_connections_style())

    # draw crosshair on face
    # faces_detect = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)
    # for (x, y, w, h) in faces_detect:
    #     # cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), thickness=4)
    #     cv.circle(frame, (int(x + w / 2), int(y + h / 5)), int(h / 6), (0, 0, 255), thickness=4)
    #     cv.line(frame, (int(x + w / 3), int(y + h / 5)), (int(x + 2 * w / 3), int(y + h / 5)), (0, 0, 255),
    #             thickness=4)
    #     cv.line(frame, (int(x + w / 2), int(y + h / 2.8)), (int(x + w / 2), int(y + h / 20)), (0, 0, 255),
    #             thickness=4)

    cv.imshow('Camera Capture', frame)
    # close program by pressing 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # readKey = cv2.waitKey(1)
    # if readKey == ord('b'):
    #     cv2.imwrite('screenshot_now.jpg', frame)
    #     print('Zrzut zrobiony')
    # elif readKey == ord('t'):
    #     cv2.circle(frame, (320,240), 30, (255,0,128), 5)
    #     cv2.imwrite('screenshot_now.jpg', frame)
    #     print('Rysuj kropke')

cap.release()
cv.destroyAllWindows()
