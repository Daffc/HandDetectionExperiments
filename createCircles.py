import cv2
import time
import math
from HandTracking import HandDetector




def main():

    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = HandDetector(detectionCon=0.7)

    cap.set(3, 1280)
    cap.set(4, 720)
    
    while True:
        success, img = cap.read()
        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=False)

        # Check if at least one list of landmarks was detected.
        if lmList:

            # For each 
            for handLmList in lmList:
                dst = math.dist([handLmList[4][1],handLmList[4][2]], [handLmList[8][1],handLmList[8][2]])
                
                mx = (handLmList[0][1] + handLmList[5][1] + handLmList[9][1] + handLmList[13][1] + handLmList[17][1]) / 5
                my = (handLmList[0][2] + handLmList[5][2] + handLmList[9][2] + handLmList[13][2] + handLmList[17][2]) / 5
                
                cv2.circle(img, (int(mx), int(my)), int((dst / 2)), (255,255,255), cv2.FILLED)

        # Calculating and displaing frames
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10,55), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()