import torch

# Download do YOLO e modelo
model = torch.hub.load('WongKinYiu/yolov7', 'yolov7')

img = "inference/gta_v_tunel_640.jpeg"
results = model(img)
results.print()
print(results.xyxy[0])
