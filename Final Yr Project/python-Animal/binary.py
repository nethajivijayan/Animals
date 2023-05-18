import cv2
import random
from time import sleep
cap = cv2.VideoCapture(0)
x=0
y=0
random.seed(5)


while True:
    ret, img = cap.read()

    while x< 1920:
        while y<1080:
            y=y+3
            r1 = random.randint(5, 15)
            if r1 < 7 :
                cv2.circle(img, (x,y), 1, (255, 0, 0), 1, cv2.LINE_8, 0)
            else:
                cv2.circle(img, (x,y), 1, (0,255, 0), 1, cv2.LINE_8, 0)
        x=x+3
        y=0
    x=0
    y=0
    cv2.imshow("Binary",img)
    sleep(1)
    cv2.destroyAllWindows()
