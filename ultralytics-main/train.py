from ultralytics import YOLO


model = YOLO("yolov8n.yaml")  # 从头开始构建新模型
model = YOLO("yolov8n.pt")  # 加载预训练的YOLOv8n模型（建议用于训练）


# 使用模型
model.train(
    data="coco128.yaml", 
    epochs=3, 
    device="cuda", 
    )  # 训练模型
# metrics = model.val() # 在验证集上评估模型性能
results = model('https://ultralytics.com/images/bus.jpg') #对图像进行预测
# success = model.export(format="onnx") #将模型导出为ONNX格式 