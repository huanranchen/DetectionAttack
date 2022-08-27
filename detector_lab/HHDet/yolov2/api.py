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
        self.eval()

    def detect_test(self, batch_tensor):
        detections_with_grad = self.detector(batch_tensor)
        return detections_with_grad

    def __call__(self, batch_tensor, **kwargs):
        detections_with_grad = self.detector(batch_tensor)  # torch.tensor([1, num, classes_num+4+1])
        # x1, y1, x2, y2, det_conf, cls_max_conf, cls_max_id
        all_boxes, obj_confs, cls_max_ids = get_region_boxes(detections_with_grad, self.conf_thres,
                                            self.detector.num_classes, self.detector.anchors,
                                            self.detector.num_anchors)
        obj_confs = obj_confs.view(batch_tensor.size(0), -1)
        cls_max_ids = cls_max_ids.view(batch_tensor.size(0), -1)

        bbox_array = []
        for boxes in all_boxes:
            # if len(boxes) == 0:
            #     bbox_array.append(torch.tensor([]).to(self.device))
            #     continue
            boxes = torch.FloatTensor(boxes).to(self.device)
            if len(boxes):
                boxes[:, :4] = torch.clamp(boxes[:, :4], min=0, max=1)
            bbox_array.append(boxes)
        bbox_array = inter_nms(bbox_array, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        # print(bbox_array)
        output = {'bbox_array': bbox_array, 'obj_confs': obj_confs, "cls_max_ids": cls_max_ids}
        return output