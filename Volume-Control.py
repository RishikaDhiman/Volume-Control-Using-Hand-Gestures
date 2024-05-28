import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import math
import pycaw
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
volume.SetMasterVolumeLevel(0, None)
minVol = volRange[0]
maxVol = volRange[1]
print("volrange",volRange, " minVol",minVol, " maxVol",maxVol)

# landmarks for hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

wCam , hCam = (640, 480)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

vol = 0
volPer = 0
volBar = 400

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while True:
        success, img = cap.read()

        # converting to rgb as mediapipe work with rgb
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Print and draw hand landmarks if any are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # print(hand_landmarks)
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the coordinates of the index and thumb finger tips
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Convert normalized coordinates to pixel values
                h, w, c = image.shape
                index_tip_x, index_tip_y = int(index_tip.x * w), int(index_tip.y * h)
                thumb_tip_x, thumb_tip_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                cx, cy = (index_tip_x+thumb_tip_x)//2, (index_tip_y+thumb_tip_y)//2

                # print(index_tip_x, index_tip_y)
                # print(thumb_tip_x, thumb_tip_y)



                cv2.circle(image, (index_tip_x, index_tip_y), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(image, (thumb_tip_x, thumb_tip_y), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(image, (index_tip_x, index_tip_y),(thumb_tip_x, thumb_tip_y), (255, 0, 255), 3)

                length = math.hypot(thumb_tip_x - index_tip_x, thumb_tip_y - index_tip_y)
                # print(length)

                if length < 20:
                    cv2.circle(image, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

                # print(length)

        #       hand range min = 40, max = 180
        #       volRange min = -65, max = 0

                vol = np.interp(length, [40, 180], [minVol, maxVol])
                volBar = np.interp(length, [40, 180], [400, 150])
                volPer = np.interp(length, [40, 180], [0, 100])

                print(length, vol)
                volume.SetMasterVolumeLevel(vol, None)

        cv2.rectangle(image, (50, 150), (85, 400), (0,255,0),3)
        cv2.rectangle(image, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)

        cv2.putText(image, f'{int(volPer)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255,0), 3)

        cv2.imshow("Img",image)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()


# print(results.multi_hand_landmarks)
