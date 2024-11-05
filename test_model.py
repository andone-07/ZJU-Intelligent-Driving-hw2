from ultralytics import YOLO

CFG = r'D://Desktop//ultralytics-main//ultralytics//cfg//models//v8//yolov8_EMA.yaml'
SOURCE = r'D://Desktop//ultralytics-main//ultralytics//assets//bus.jpg'

def test_model_forward():
    model = YOLO(CFG)
    model(SOURCE)