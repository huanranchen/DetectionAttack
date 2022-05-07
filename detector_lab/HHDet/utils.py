from .yolov3.api import HHYolov3
from .yolov4.api import HHYolov4
from .yolov5.api import HHYolov5
from .military_faster_rcnn.api import FRCNN as Military_FasterRCNN
from .military_yolox.api import YOLO as Military_YOLOX



def init_detector(detector_name):
    if detector_name == "YOLOV3":
        detector = HHYolov3(name="YOLOV3")
        detector.load(detector_config_file='./detector_lab/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3.cfg',
                      detector_weights='./detector_lab/HHDet/yolov3/PyTorch_YOLOv3/weights/yolov3.weights')

    if detector_name == "YOLOV4":
        detector = HHYolov4(name="YOLOV4")
        detector.load('./detector_lab/HHDet/yolov4/Pytorch_YOLOv4/cfg/yolov4.cfg',
                      './detector_lab/HHDet/yolov4/Pytorch_YOLOv4/weight/yolov4.weights',
                      './detector_lab/HHDet/yolov4/Pytorch_YOLOv4')

    if detector_name == "YOLOV5":
        detector = HHYolov5(name="YOLOV5")
        detector.load('./detector_lab/HHDet/yolov5/yolov5/weight/yolov5s.pt')

    if detector_name == "Military_FasterRCNN":
        detector = Military_FasterRCNN(name="Military_FasterRCNN")
        detector.load(
            detector_weights='./detector_lab/HHDet/military_faster_rcnn/faster_rcnn/logs/ep300-loss0.701-val_loss1.057.pth',
            classes_path='./detector_lab/HHDet/military_faster_rcnn/faster_rcnn/model_data/car_classes.txt')

    if detector_name == "Military_YOLOX":
        detector = Military_YOLOX(name="Military_YOLOX")
        detector.load(
            detector_weights='./detector_lab/HHDet/military_yolox/yolox/logs/ep1155-loss1.474-val_loss1.452.pth',
            classes_path='./detector_lab/HHDet/military_yolox/yolox/model_data/car_classes.txt')

    return detector