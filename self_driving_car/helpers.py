import cv2
import numpy as np


def roi(image, vertices):
  mask = np.zeros_like(image)
  color = (255, 255, 255) if image.ndim == 3 else 255
  cv2.fillPoly(mask, [vertices], color)
  masked = cv2.bitwise_and(image, mask)
  return masked


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
  '''
  Plots one bounding box on image img

  This function was extracted from [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7/blob/13594cf6d42bc3a49ff226570aa77b6bf7615f7f/utils/plots.py).
  '''
  tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
  color = color or [np.random.randint(0, 255) for _ in range(3)]
  c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
  cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
  if label:
    tf = max(tl - 1, 1)
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)