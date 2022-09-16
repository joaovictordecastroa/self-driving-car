import cv2
import numpy as np
import mss

bounding_box = {'top': 100, 'left': 0, 'width': 400, 'height': 300}

sct = mss()

while True:
    sct_img = sct.grab(bounding_box)
    cv2.imshow('screen', np.array(sct_img))

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break