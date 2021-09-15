import cv2
import time
import math
from HandTracking import HandDetector
import random



class Circle():
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color

    def update(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color

    def draw(self, img):
        cv2.circle(img, (self.x, self.y), self.radius, self.color, cv2.FILLED)

    def insideCircle(self, x, y):
        distance = math.dist([x, y], [self.x, self.y])        
        
        if distance <= self.radius:
            return True

        return False

def main():

    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = HandDetector(detectionCon=0.7)

    cap.set(3, 1280)
    cap.set(4, 720)

    circles = []

    for i in range(5):
        circles.append(Circle(random.randrange(1280), random.randrange(720), 30, (0, 255, 0)))

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        img = detector.findHands(img, draw=True)
        lmList = detector.findPosition(img, draw=False)

        
        # Check if at least one list of landmarks was detected.
        if lmList:

            # For each hand
            for handLmList in lmList:

                # Calculating the distance and point betwen the tips of index and thumb
                Tipsdst = math.dist([handLmList[4][1],handLmList[4][2]], [handLmList[8][1],handLmList[8][2]])
                cx = int((handLmList[4][1] + handLmList[8][1]) / 2)
                cy = int((handLmList[4][2] + handLmList[8][2]) / 2)
                
                # for each circle, check if it has been caught, updaing the center.
                for circle in circles:
                    if(circle.insideCircle(cx, cy) and Tipsdst < 70):
                        circle.x = cx
                        circle.y = cy

        # Draw all circles.
        for circle in circles:       
            circle.draw(img)

        # Calculating and displaing frames
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10,55), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()