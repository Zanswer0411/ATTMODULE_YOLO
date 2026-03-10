from jinja2.optimizer import optimize
from torch.cuda import device

from ultralytics import YOLO

if __name__ == "__main__":
    # 加载YOLOv8模型
    model = YOLO("weights/yolov8n.pt")  # 替换为你的模型路径（如yolov8s.pt、自定义训练的模型）

    # 导出兼容OpenCV DNN的ONNX模型（关键参数）
    path = model.export(
        format="onnx",
        imgsz=(640, 640), #设置输入图像的尺寸
        keras=False, #Keras模式
        optimize=False, #用于在导出为TorchScript格式时进行模型优化
        half=False,  # FP16精度
        int8=False, #INT8量化
        dynamic=False, #动态输入尺寸
        simplify=True,  # 简化模型，移除冗余节点（核心！）
        opset=None,  # opset版本,None使用最新版本
        workspace=4.0, #为TensorRT优化设置最大工作区大小（GiB）
        nms=False, #NMS(非极大值抑制)
        batch=1,  # 固定batch=1，OpenCV DNN通常只支持单张推理
        device="0" #指定导出设备为CPU或GPU
    )