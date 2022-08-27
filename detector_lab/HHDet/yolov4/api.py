import sys

import numpy as np
import torch

from .Pytorch_YOLOv4.tool.utils import *
from .Pytorch_YOLOv4.tool.torch_utils import *
from .Pytorch_YOLOv4.tool.darknet2pytorch import Darknet

from ...DetectorBase import DetectorBase


class HHYolov4(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_size=416,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)

    def requires_grad_(self, state: bool):
        assert self.detector
        self.detector.models.requires_grad_(state)

    def load(self, model_weights, detector_config_file=None, data_config_path=None):
        if self.cfg.PERTURB.GATE == 'shake_drop':
            from .Pytorch_YOLOv4.tool.darknet_shakedrop import DarknetShakedrop
            tmp = detector_config_file.split('/')
            detector_cfg = self.cfg.PERTURB.SHAKE_DROP.MODEL_CONFIG
            print('Self ensemble! Shake drop model cfg :', detector_config_file)
            tmp[-1] = detector_cfg
            self.detector = DarknetShakedrop('/'.join(tmp)).to(self.device)

            self.clean_model = Darknet(detector_config_file).to(self.device)
            self.clean_model.load_weights(model_weights)
            self.clean_model.eval()
            self.clean_model.models.requires_grad_(False)
        else:
            self.detector = Darknet(detector_config_file).to(self.device)

        self.detector.load_weights(model_weights)
        self.eval()

    def __call__(self, batch_tensor, clean_model=False):
        if clean_model and hasattr(self, 'clean_model'):
            detections_with_grad = self.clean_model(batch_tensor)
        else:
            detections_with_grad = self.detector(batch_tensor)

        bbox_array = post_processing(batch_tensor, self.conf_thres, self.iou_thres, detections_with_grad)
        print(bbox_array)
        for i, pred in enumerate(bbox_array):
            print(pred)
            if len(pred) == 0:
                bbox_array.append(torch.tensor([]).to(self.device))
                continue
            pred = torch.Tensor(pred).to(self.device)
            if len(pred) != 0:
                pred[:, :4] = torch.clamp(pred[:, :4], min=0, max=1)
            bbox_array[i] = pred # shape([1, 6])

        # output: [ [batch, num, 1, 4], [batch, num, num_classes] ]
        # v4's confs is the combination of obj conf & cls conf
        obj_confs = detections_with_grad[1]
        cls_max_ids = torch.argmax(obj_confs, dim=2)
        output = {'bbox_array': bbox_array, 'obj_confs': obj_confs, "cls_max_ids": cls_max_ids}
        return output
