import torch
import cv2
from helpers import plot_one_box

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