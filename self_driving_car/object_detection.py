import torch
import cv2
import numpy as np


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
  # Plots one bounding box on image img
  tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
  color = color or [np.random.randint(0, 255) for _ in range(3)]
  c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
  cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
  if label:
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
    cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


model = torch.hub.load('WongKinYiu/yolov7', 'yolov7')

img_path = "images/gta_v_tunel_640.jpeg"
image = cv2.imread(img_path)

with torch.no_grad():
  results = model(image)

results.print()

for *box, conf, cls in results.pred[0]:
  label = f'{results.names[int(cls)]} {conf:.2f}'
  plot_one_box(box, image, label=label, line_thickness=1)

print(results.xyxy[0])

cv2.imshow('Output', image)
cv2.waitKey(0)