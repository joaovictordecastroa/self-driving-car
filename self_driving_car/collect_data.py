import cv2
import numpy as np
from mss import mss
from time import time
import torch

from helpers import plot_one_box, roi
from joystick import PyvJoyXboxController


# Macros
line_detection_enabled = False
object_detection_enabled = True


assert line_detection_enabled != object_detection_enabled


def line_detection(image, vertices):
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.Canny(image, 100, 200)

    image = roi(image, vertices)

    return image


obstacles_labels = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

joy = PyvJoyXboxController([])
joy.set_axis('LT', -1)
joy.set_axis('RT', -1)

def object_detection(model, image):
    with torch.no_grad():
        results = model(image)
    
    max_size = 0

    for *box, conf, cls in results.pred[0]:
        if results.names[int(cls)] in obstacles_labels:
            size = np.sqrt(np.square(float(box[2]) - float(box[0])) + np.square(float(box[3]) - float(box[1])))

            if size > max_size:
                max_size = size

            label = f'{results.names[int(cls)]} {conf:.2f} | {size:.2f}'

            plot_one_box(box, image, label=label, line_thickness=1)
    
    print(f'max size: {max_size}')
    
    if max_size >= 70:
        joy.set_axis('LT', 1)
        joy.set_axis('RT', -1)
    elif max_size >= 40:
        joy.set_axis('LT', 0)
        joy.set_axis('RT', -1)
    else:
        joy.set_axis('LT', -1)
        joy.set_axis('RT', 0)

    return image
    

if __name__ == '__main__':
    monitor = {'left': 0, 'top': 40, 'width': 800, 'height': 600}

    scale = 1 / 4

    vertices = np.array([[0, 600], [0, 400], [200, 200], [600, 200], [800, 400], [800, 600]], np.int32)
    vertices = (vertices * scale).astype(np.int32)

    if object_detection_enabled:
        model = torch.hub.load('WongKinYiu/yolov7', 'yolov7')

    with mss() as sct:
        while True:
            t0 = time()
            
            image = np.array(sct.grab(monitor))
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

            if object_detection_enabled:
                image = object_detection(model, image)
            elif line_detection_enabled:
                image = line_detection(image, vertices)

            t1 = time()
            
            print(f'time: {t1 - t0}')

            cv2.imshow('OpenCV output', image)

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break