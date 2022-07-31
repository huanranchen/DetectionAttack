import sys

import torch
import numpy as np

from . import faster_rcnn, fasterrcnn_resnet50_fpn
from ...DetectorBase import DetectorBase
from ..utils import inter_nms


class Faster_RCNN(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_size=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)

    def load(self, model_weights=None, **args):
        kwargs = {}
        if self.input_tensor_size is not None:
            kwargs['min_size'] = self.input_tensor_size
        self.detector = fasterrcnn_resnet50_fpn(pretrained=True, **kwargs) \
            if model_weights is None else fasterrcnn_resnet50_fpn()

        self.detector = self.detector.to(self.device)
        self.detector.eval()
        self.detector.requires_grad_(False)

    def detect_img_batch_get_bbox_conf(self, batch_tensor, confs_thresh=0.5):
        shape = batch_tensor.shape[-2]
        preds, confs = self.detector(batch_tensor)

        confs_array = None
        bbox_array = []
        for ind, (pred, conf) in enumerate(zip(preds, confs)):
            nums = pred['scores'].shape[0]
            array = torch.cat((
                pred['boxes'] / shape,
                pred['scores'].view(nums, 1),
                (pred['labels'] - 1).view(nums, 1)
            ), 1).detach().cpu() if nums else torch.FloatTensor([])

            ctmp = conf[conf > confs_thresh]
            confs_array = ctmp if confs_array is None else torch.cat((confs_array, ctmp), -1)

            bbox_array.append(array)

        bbox_array = inter_nms(bbox_array, self.conf_thres, self.iou_thres)

        return bbox_array, confs_array