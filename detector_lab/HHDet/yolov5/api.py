import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path


from torch.autograd.grad_mode import F
from torch.autograd import Variable

# load from YOLOV5
from .yolov5.utils.augmentations import letterbox
from .yolov5.utils.general import non_max_suppression, scale_coords
from .yolov5.utils.plots import Annotator, colors
from .yolov5.models.experimental import attempt_load  # scoped to avoid circular import


from ...DetectorBase import DetectorBase


class HHYolov5(DetectorBase):
    def __init__(self, name, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        # detector
        super().__init__(name, device)
        self.detector = None
        self.imgsz = (640, 640)
        self.detector, self.stride, self.names, self.pt = None, None, None, None

    def load(self, model_weights, **args):
        w = str(model_weights[0] if isinstance(model_weights, list) else model_weights)
        self.detector = attempt_load(model_weights if isinstance(model_weights, list) else w, map_location=self.device, inplace=False)
        self.detector.eval()
        self.stride = max(int(self.detector.stride.max()), 32)  # model stride
        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names  # get class names

    def normalize(self, image_data):
        image_data = image_data.squeeze(0).transpose((2, 0, 1))[::-1]
        image_data = np.ascontiguousarray(image_data) / 255.
        img_tensor = torch.from_numpy(image_data).to(self.device).float()
        img_tensor = Variable(img_tensor)
        img_tensor.requires_grad = True
        return img_tensor, image_data

    def unnormalize(self, img_tensor):
        # img_tensor: tensor [1, c, h, w]
        img_numpy = img_tensor.squeeze(0).cpu().detach().numpy()
        img_numpy = np.transpose(img_numpy, (1, 2, 0))
        img_numpy *= 255
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        img_numpy_int8 = img_numpy.astype('uint8')
        return img_numpy, img_numpy_int8

    def detect_img_batch_get_bbox_conf(self, batch_tensor):
        output = self.detector(batch_tensor, augment=False, visualize=False)[0]
        # [batch, num, 1, 4] e.g., [1, 22743, 1, 4]
        box_array = output[:, :, :4].unsqueeze(2)
        # [batch, num, num_classes] e.g.,[1, 22743, 80]
        confs = output[:, :, 5:]
        return box_array, confs