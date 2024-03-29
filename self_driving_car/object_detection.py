import torch
import cv2
from helpers import plot_one_box
from math import sqrt

obstacles_labels = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

model = torch.hub.load('WongKinYiu/yolov7', 'yolov7')

img_path = "images/gta_v_tunel_640.jpeg"
image = cv2.imread(img_path)

with torch.no_grad():
  results = model(image)

results.print()

for *box, conf, cls in results.pred[0]:
  diagonal_length = sqrt((box[2] - box[0])**2 + (box[3] - box[1])**2)
  label = f'{results.names[int(cls)]} {conf:.2f} | {diagonal_length:.2f}'
  if results.names[int(cls)] in obstacles_labels:
    plot_one_box(box, image, label=label, line_thickness=1)

print(results.xyxy[0])

cv2.imshow('Output', image)
cv2.waitKey(0)