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

        if lmList:

            # Chack fist Hand
            if 0 < len(lmList):
                dst = math.dist([lmList[4][1],lmList[4][2]], [lmList[8][1],lmList[8][2]])
                
                mx = (lmList[0][1] + lmList[5][1] + lmList[9][1] + lmList[13][1] + lmList[17][1]) / 5
                my = (lmList[0][2] + lmList[5][2] + lmList[9][2] + lmList[13][2] + lmList[17][2]) / 5
                
                cv2.putText(img, str(dst), (10, 95), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                cv2.circle(img, (int(mx), int(my)), int((dst / 2)), (255,255,255), cv2.FILLED)
            
            #Check second Hand
            if 21 < len(lmList):
                dst = math.dist([lmList[25][1],lmList[25][2]], [lmList[29][1],lmList[29][2]])
                
                mx = (lmList[25][1] + lmList[29][1]) / 2
                my = (lmList[25][2] + lmList[29][2]) / 2
                
                cv2.putText(img, str(dst), (10, 135), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
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