import cv2
import time
import math
from HandTracking import HandDetector
import random



class Circle():
    def __init__(self, x, y, radius, color, velocity):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.velocity = velocity

    def move(self):
        self.y += self.velocity

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
    points = 0

    cWidth = 1280
    cHeight = 720

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = HandDetector(detectionCon=0.7)

    cap.set(3, cWidth)
    cap.set(4, cHeight)

    circles = []

    lpTime = time.time()
    spawnInterval = random.randrange(1, 3)

    endGame = False

    while not endGame:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # Detecting hands and recovering landmarks.
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
                    if(circle.insideCircle(cx, cy) and Tipsdst < 50):
                        circles.remove(circle)
                        del circle
                        points += 1
                    

        # Updating circles position
        for circle in circles:
            # Move circle to the next position
            circle.move()
            
            # If center of the circle transpasses height, end of game.
            if (circle.y >= cHeight):
                circles.remove(circle)
                del circle
                endGame = True

        # Draw all circles.
        for circle in circles:       
            circle.draw(img)

        # Calculating and displaing frames
        cTime = time.time()
        dTime = cTime-pTime
        fps = 1/(cTime-pTime)
        pTime = cTime

        # If reach spaw time interval, spaw a new circle, redefining new spaw interval
        if ((cTime - lpTime) >= spawnInterval):
            # Defining circle velocity according to the current obtained points (increase dificulty).
            newVelocity = random.randrange(1, 3) + int((2**(points/20)) - 1)
            circles.append(Circle(random.randrange(cWidth - 100) + 50, 10, 30, (0, 255, 0), newVelocity))
            lpTime = time.time()
            spawnInterval = random.randrange(1, 3)

        #Displaying FPS and points.
        cv2.putText(img, str(int(fps)), (10,55), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.putText(img, str(int(points)), (10,95), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
        
        cv2.imshow("Image", img)
        cv2.waitKey(1)

    # Display final poits.
    print("RESULT:", points)
    

if __name__ == "__main__":
    main()