import torch
import cv2 as cv
from ultralytics import YOLO

model = YOLO('weights/yolov8n.pt')  # 加载模型


img = cv.imread('ultralytics/assets/bus.jpg')  # 读取输入图像
results = model(img)  # 进行推理
for r in results:
    r.save(filename='runs/detect/result.jpg')
    r.print()