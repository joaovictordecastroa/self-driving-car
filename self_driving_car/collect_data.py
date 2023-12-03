import cv2
import numpy as np
from mss import mss
import torch
from helpers import plot_one_box, roi
from joystick import PyvJoyXboxController

from lane_detection import LaneDetection

MAX_FPS = 15
MIN_FRAME_TIME = 1 / MAX_FPS
count = 0

# Macros
line_detection_enabled = True
object_detection_enabled = False
lane_detection = LaneDetection()

assert line_detection_enabled != object_detection_enabled

obstacles_labels = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

joy = PyvJoyXboxController([])
joy.set_axis('LT', -1)
joy.set_axis('RT', -1)
joy.set_axis('LS_X', 0)


def object_detection(model, image):
    image = roi(image, vertices_obj_det)

    with torch.no_grad():
        results = model(image)

    max_size = 0

    for *box, conf, cls in results.pred[0]:
        if results.names[int(cls)] in obstacles_labels:
            size = np.sqrt(np.square(
                float(box[2]) - float(box[0])) + np.square(float(box[3]) - float(box[1])))

            if size > max_size:
                max_size = size

            label = f'{results.names[int(cls)]} {conf:.2f} | {size:.2f}'

            plot_one_box(box, image, label=label, line_thickness=1)

    print(f'max size: {max_size}')

    if max_size >= 60:
        joy.set_button('RB', 1)
        joy.set_axis('LT', 1)
        joy.set_axis('RT', -1)
    elif max_size >= 50:
        joy.set_button('RB', 0)
        joy.set_axis('LT', -0.25)
        joy.set_axis('RT', -0.2)
    elif max_size >= 40:
        joy.set_button('RB', 0)
        joy.set_axis('LT', -0.3)
        joy.set_axis('RT', -0.2)
    elif max_size >= 30:
        joy.set_button('RB', 0)
        joy.set_axis('LT', -0.6)
        joy.set_axis('RT', -0.2)
    elif max_size >= 20:
        joy.set_button('RB', 0)
        joy.set_axis('LT', -0.8)
        joy.set_axis('RT', -0.2)
    elif max_size >= 10:
        joy.set_button('RB', 0)
        joy.set_axis('LT', -1)
        joy.set_axis('RT', -0.2)
    else:
        joy.set_button('RB', 0)
        joy.set_axis('LT', -1)
        joy.set_axis('RT', 0)

    return image


def lane_direction(lines, thetas):
    if lines is None :
        print('Vai reto1')
        joy.set_axis('LS_X', 0)
    elif lines[1] is not None:
        print(f'Vai reto2')
        joy.set_axis('LS_X', 0)
    else:
        if lines[0][0][0] < 300:
            print('Deve ir para a direita')
            joy.set_axis('LS_X', 0.35)
        else:
            print('Deve ir para a esquerda')
            joy.set_axis('LS_X', -0.35)
        print(thetas)
    print('\n')


if __name__ == '__main__':
    monitor = {'left': 0, 'top': 40, 'width': 800, 'height': 600}

    scale = 1 / 2

    '''vertices_obj_det = np.array(
        [[300, 600], [300, 0], [500, 0], [500, 600]], np.int32)'''  # Usado para fazer a calibração dos parâmetros do lane detection, não é para ser rodado junto com o jogo
    vertices_obj_det = np.array([[0, 600], [0, 500], [266, 375], [
        532, 250], [1000, 500], [1000, 600]], np.int32)
    vertices_obj_det = (vertices_obj_det * scale).astype(np.int32)

    if object_detection_enabled:
        model = torch.hub.load('WongKinYiu/yolov7', 'yolov7')

    with mss() as sct:
        while True:
            #print(count)

            # screen capture
            image = np.array(sct.grab(monitor))
            min_threshold = 111
            max_threshold = 71
            max_lane_gap = 50
            threshold = 180
            rho = 1

            if object_detection_enabled:
                image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
                image = object_detection(model, image)

            if line_detection_enabled:
                count += 1
                image, lines, thetas = lane_detection.get_lanes(image, min_threshold, max_threshold, max_lane_gap,
                                                                threshold, rho)
                if count == 30:
                    lane_direction(lines, thetas)
                    count = 0

            cv2.imshow('OpenCV output', image)

            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break
