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
import time
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
ret, frame = cap.read()
cTime = 0
pTime = 0
height, width, channel = frame.shape

ALLOWABLE_MOVE_RANGE = 10

RightHandLandMarksPositionsDictionary = {'nadgarstek': [0, 0],
                                'kciuk': [0, 0],
                                'wskazujacy':[0, 0],
                                'srodkowy':[0, 0],
                                'serdeczny':[0, 0],
                                'maly':[0, 0],
                                }

LeftHandLandMarksPositionsDictionary = {'nadgarstek': [0, 0],
                                'kciuk': [0, 0],
                                'wskazujacy': [0, 0],
                                'srodkowy': [0, 0],
                                'serdeczny': [0, 0],
                                'maly': [0, 0]
                                }



def logPosition(index, pixLocX, pixLocY, whichHand):
    partOfHandName = ''
    if index == 0:
        partOfHandName = 'nadgarstek'
        print(index, 'nadgarstek = ', pixLocX, pixLocY)
    elif index == 4:
        partOfHandName = 'kciuk'
        print(index, 'kciuk = ', pixLocX, pixLocY)
    elif index == 8:
        partOfHandName = 'wskazujacy'
        print(index, 'wskazujacy = ', pixLocX, pixLocY)
    elif index == 12:
        partOfHandName = 'srodkowy'
        print(index, 'srodkowy = ', pixLocX, pixLocY)
    elif index == 16:
        partOfHandName = 'serdeczny'
        print(index, 'serdeczny = ', pixLocX, pixLocY)
    elif index == 20:
        partOfHandName = 'maly'
        print(index, 'maly = ', pixLocX, pixLocY)
    if whichHand == 'L':
        LeftHandLandMarksPositionsDictionary[partOfHandName] = [pixLocX, pixLocY]
    else:
        RightHandLandMarksPositionsDictionary[partOfHandName] = [pixLocX, pixLocY]

def drawCrosshair(contours, frame):

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

def drawAndCalculateHands(frame):
    # convert colours captured from camera to RGB
    imgRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            label = hand_handedness.classification[0].label
            if label == 'Right':
                print('\n', idx, 'Right hand: ')
                for id, landmark in enumerate(results.multi_hand_landmarks[idx].landmark):
                    pixelLocationX = int(landmark.x * width)
                    pixelLocationY = int(landmark.y * height)
                    logPosition(id, pixelLocationX, pixelLocationY, 'R')
                    #mpDraw.draw_landmarks(frame, results.multi_hand_landmarks[idx], mpHands.HAND_CONNECTIONS)
            elif label == 'Left':
                print('\n', idx, 'Left hand: ')
                for id, landmark in enumerate(results.multi_hand_landmarks[idx].landmark):
                    pixelLocationX = int(landmark.x * width)
                    pixelLocationY = int(landmark.y * height)
                    logPosition(id, pixelLocationX, pixelLocationY, 'L')
                    #mpDraw.draw_landmarks(frame, results.multi_hand_landmarks[idx], mpHands.HAND_CONNECTIONS)

while cap.isOpened():
    # captureVideo()
    ret, frame = cap.read()
    _, frame_2 = cap.read()
    cv.imshow('Camera Capture', frame)

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

    drawCrosshair(contours, frame)
    drawAndCalculateHands(frame)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(frame, str(int(fps)), (10,70), cv.FONT_ITALIC, 3, (255,255,0), 3)

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
    readKey = cv.waitKey(1)

    if readKey == ord('b'):
        cv.imwrite('screenshot_now.jpg', frame)
        print('Zrzut zrobiony')
    elif readKey == ord('t'):
        cv.circle(frame, (320,240), 30, (255,0,128), 5)
        #cv2.imwrite('screenshot_now.jpg', frame)
        print('Rysuj kropke')
    # close program by pressing 'q' key
    elif readKey == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
