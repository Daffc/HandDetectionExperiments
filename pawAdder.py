import cv2
import time
import math
import numpy as np
from PIL import Image
from numpy.lib.function_base import average
from HandTracking import HandDetector


# Return angle in degrees resulting be the angle between the line define by landmarks (lm1 and lm2) relative to the x-axis.
def getLandmarksAngle(lm1: list, lm2: list):
    radians = math.atan2(lm1[2] - lm2[2], lm1[1] - lm2[1])
    degrees = radians * 180 / np.pi
    return degrees

# Return new image redimensioned and rotated according to the parameters.
def adjustPaw(img: Image, dimensions: tuple, angle: float):

    resizedPaw = img.resize(dimensions)
    rotatedPaw = resizedPaw.rotate(angle, expand=True)

    return rotatedPaw

# Returns centroid described in landmarks list.
def getCentroid(landmarks: list):    
    averages = list(map(average, zip(*landmarks)))
    cX = averages[1]
    cY = averages[2]
    return cX, cY

    
def main():

    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    imgPaw = Image.open(r"data/paw.png")

    detector = HandDetector(detectionCon=0.7)

    cap.set(3, 1280)
    cap.set(4, 720)
    
    while True:
        _ , frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = detector.findHands(frame, draw=True)
        lmList = detector.findPosition(frame, draw=False)

        # Check if at least one list of landmarks was detected.
        if lmList:
            
            # For each hand
            for handLmList in lmList:
                # Getting distance betwen hand base and middle finger tip.
                hDst = int(math.dist([handLmList[0][1], handLmList[0][2]], [handLmList[12][1],handLmList[12][2]]))

                # Recovering centroid of selected landmarks.
                cX, cY = getCentroid([handLmList[0], handLmList[12]])
                
                # Recovering the angle between the line defined by the selected landmarks and the x axis.
                angle = getLandmarksAngle(handLmList[0], handLmList[9])

                # Adjusting angle as so vertical line represents 0 degrees.
                vAngle = -(angle - 90)

                # Generating new paw image, adjusted according to the landmarks variations.
                newPawImg = adjustPaw(imgPaw, (hDst, hDst), vAngle)

                # Defining certer position for 'newPawImg', so as it maches with the previous calculated centroid.
                newCenter = (int(cX - (newPawImg.width/ 2)), int(cY - (newPawImg.height/ 2)))
                

                # Pasting 'newPawImg' in frame.
                pFrame = Image.fromarray(frame)
                pFrame.paste(newPawImg, box = newCenter, mask=newPawImg)
                frame = np.array(pFrame)

        # Calculating and displaing frames
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(frame, str(int(fps)), (10,55), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow("Image", frame)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()