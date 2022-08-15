import sys

import torch
import numpy as np

from .yolov2.darknet import Darknet
from .yolov2.utils import get_region_boxes, inter_nms
from ...DetectorBase import DetectorBase


class HHYolov2(DetectorBase):
    def __init__(self,
                 name, cfg,
                 input_tensor_size=412,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)

    def load(self, model_weights, detector_config_file=None):
        print(detector_config_file)
        self.detector = Darknet(detector_config_file).to(self.device)
        self.detector.load_weights(model_weights)
        self.detector.eval()
        self.detector.requires_grad_(False)

    def detect_test(self, batch_tensor):
        detections_with_grad = self.detector(batch_tensor)
        return detections_with_grad

    def detect_img_batch_get_bbox_conf(self, batch_tensor, confs_thresh=0.5):
        detections_with_grad = self.detector(batch_tensor)  # torch.tensor([1, num, classes_num+4+1])
        # print(detections_with_grad.shape, detections_with_grad.dim())

        # x1, y1, x2, y2, det_conf, cls_max_conf, cls_max_id
        all_boxes, confs = get_region_boxes(detections_with_grad, self.conf_thres, self.detector.num_classes,
                         self.detector.anchors, self.detector.num_anchors)
        confs = confs[confs > confs_thresh]
        bbox_array = []
        for boxes in all_boxes:
            if len(boxes) == 0:
                bbox_array.append(np.array([]))
                continue
            boxes = torch.FloatTensor(boxes)
            boxes[:, :4] = torch.clamp(boxes[:, :4], min=0, max=1)
            bbox_array.append(boxes)
        bbox_array = inter_nms(bbox_array, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        # print(bbox_array)
        # sys.exit()
        return bbox_array, confs