import cv2
import numpy as np
from mss import mss
from time import time

monitor = {'left': 0, 'top': 40, 'width': 800, 'height': 600}

with mss() as sct:
    while True:
        t0 = time()
        
        image = np.array(sct.grab(monitor))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        image = cv2.resize(image, (200, 150))

        image = roi(image, vertices)

        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.Canny(image, 100, 200)
        
        t1 = time()
        
        print(f'time: {t1 - t0}')

        cv2.imshow('OpenCV output', image)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break