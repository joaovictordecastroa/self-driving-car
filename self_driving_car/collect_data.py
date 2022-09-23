import cv2
import numpy as np
from mss import mss
from time import time

monitor = {'left': 0, 'top': 40, 'width': 800, 'height': 600}

scale = 1 / 4

vertices = np.array([[0, 600], [0, 400], [200, 200], [600, 200], [800, 400], [800, 600]], np.int32)
vertices = (vertices * scale).astype(np.int32)

def roi(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [vertices], 255)
    masked = cv2.bitwise_and(image, mask)
    return masked

with mss() as sct:
    while True:
        t0 = time()
        
        image = np.array(sct.grab(monitor))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.Canny(image, 100, 200)

        image = roi(image, vertices)

        t1 = time()
        
        print(f'time: {t1 - t0}')

        cv2.imshow('OpenCV output', image)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break