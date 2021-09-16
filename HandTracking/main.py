import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    
    # Identify hands in 'img'
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks :
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    # Get list of landmarks (id, x, y) for each detected hand.
    def findPosition(self, img, draw=True):
        
        HandLmList = []
        if self.results.multi_hand_landmarks :
            for hand in self.results.multi_hand_landmarks:
                handList = []
                for id, lm in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    handList.append([id, cx, cy])
                    
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (0,255,0), cv2.FILLED)
                HandLmList.append(handList)

        return HandLmList


def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = HandDetector(detectionCon=0.7)

    while True:
        success, img = cap.read()
        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=False)

        if lmList:
            print(lmList[0])

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)

    
if __name__ == "__main__":
    main()