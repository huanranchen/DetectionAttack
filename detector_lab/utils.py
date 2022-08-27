import numpy as np
import torchvision
import torch

from .HHDet.yolov2.api import HHYolov2
from .HHDet.yolov3.api import HHYolov3
from .HHDet.yolov4.api import HHYolov4
from .HHDet.yolov5.api import HHYolov5
# from .HHDet.ssd import HHSSD
# from .HHDet.faster_rcnn.api import Faster_RCNN
from .torchDet.faster_rcnn import Faster_RCNN
# from .torchDet import Faster_RCNN
from .torchDet.ssd import TorchSSD

import sys, os
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

model_map = {
    'yolov3': {
        'model': HHYolov3,
        'cfg': os.path.join(PROJECT_DIR, 'detector_lab/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3.cfg'),
        'weight': os.path.join(PROJECT_DIR, 'detector_lab/HHDet/yolov3/PyTorch_YOLOv3/weights/yolov3.weights')
    },
    'yolov3-tiny': {
        'model': HHYolov3,
        'cfg': os.path.join(PROJECT_DIR, 'detector_lab/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3-tiny.cfg'),
        'weight': os.path.join(PROJECT_DIR, 'detector_lab/weights/yolov3-tiny.weights')
    },
    'faster_rcnn': {
        'model': Faster_RCNN,
        'backbone': 'resnet', # resnet, vgg16
        'weight': None # pytroch pretrained weights
        #'weight': os.path.join(PROJECT_DIR, 'detector_lab/HHDet/faster_rcnn/pytorch_faster_rcnn/weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth')
    }
}


def init_detectors(detector_names, cfg=None):
    detectors = []
    for detector_name in detector_names:
        detector = init_detector(detector_name, cfg.DETECTOR)
        detectors.append(detector)
    return detectors


def init_detector(detector_name, cfg):
    detector = None
    detector_name = detector_name.lower()

    if detector_name == "yolov2":
        detector = HHYolov2(name=detector_name, cfg=cfg)
        detector.load(
            model_weights=os.path.join(PROJECT_DIR, 'detector_lab/HHDet/yolov2/yolov2/weights/yolo.weights'),
            detector_config_file=os.path.join(PROJECT_DIR, 'detector_lab/HHDet/yolov2/yolov2/config/yolo.cfg'),
        )

    elif detector_name == "yolov3":
        detector = HHYolov3(name=detector_name, cfg=cfg)
        model_cfg = 'yolov3.cfg'
        detector.load(
            detector_config_file=os.path.join(PROJECT_DIR, f'detector_lab/HHDet/yolov3/PyTorch_YOLOv3/config/{model_cfg}'),
            model_weights=os.path.join(PROJECT_DIR, 'detector_lab/HHDet/yolov3/PyTorch_YOLOv3/weights/yolov3.weights'),
        )

    elif detector_name == "yolov3-tiny":
        detector = HHYolov3(name=detector_name, cfg=cfg)
        detector.load(
            detector_config_file=os.path.join(PROJECT_DIR, 'detector_lab/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3-tiny.cfg'),
            model_weights=os.path.join(PROJECT_DIR, 'detector_lab/weights/yolov3-tiny.weights'))

    elif detector_name == "yolov4-tiny":
        detector = HHYolov4(name=detector_name, cfg=cfg)
        detector.load(
            detector_config_file=os.path.join(PROJECT_DIR, 'detector_lab/HHDet/yolov4/Pytorch_YOLOv4/cfg/yolov4-tiny.cfg'),
            model_weights=os.path.join(PROJECT_DIR, 'detector_lab/weights/yolov4-tiny.weights'))

    elif detector_name == "yolov4":
        detector = HHYolov4(name=detector_name, cfg=cfg)
        model_cfg = 'yolov4.cfg'
        detector.load(
            detector_config_file=os.path.join(PROJECT_DIR, f'detector_lab/HHDet/yolov4/Pytorch_YOLOv4/cfg/{model_cfg}'),
            model_weights=os.path.join(PROJECT_DIR, 'detector_lab/HHDet/yolov4/Pytorch_YOLOv4/weight/yolov4.weights')
        )

    elif detector_name == "yolov5":
        # model = HHYolov5
        detector = HHYolov5(name=detector_name, cfg=cfg)
        detector.load(
            model_weights=os.path.join(PROJECT_DIR, 'detector_lab/HHDet/yolov5/yolov5/weight/yolov5s.pt'))

    elif detector_name == "ssd":
        # pass
        detector = TorchSSD(name=detector_name, cfg=cfg)
        detector.load()
        # detector.load('./checkpoints/ssd300_coco_20210803_015428-d231a06e.pth')
        # detector.load(os.path.join(PROJECT_DIR, 'detector_lab/HHDet/ssd/ssd_pytorch/weights/vgg16_reducedfc.pth'))

    elif detector_name == "faster_rcnn":
        detector = Faster_RCNN(detector_name, cfg)
        detector.load()

    return detector


def inter_nms(all_predictions, conf_thres=0.25, iou_thres=0.45):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    max_det = 300  # maximum number of detections per image
    out = []
    for predictions in all_predictions:
        # for each img in batch
        # print('pred', predictions.shape)
        if not predictions.shape[0]:
            out.append(predictions)
            continue
        if type(predictions) is np.ndarray:
            predictions = torch.from_numpy(predictions)
        # print(predictions.shape[0])
        try:
            scores = predictions[:, 4]
        except Exception as e:
            print(predictions.shape)
            assert 0==1
        i = scores > conf_thres

        # filter with conf threshhold
        boxes = predictions[i, :4]
        scores = scores[i]

        # filter with iou threshhold
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        # print('i', predictions[i].shape)
        out.append(predictions[i])
    return out


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names