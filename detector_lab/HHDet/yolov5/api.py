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
    def __init__(self, name, cfg,
                 input_tensor_size=640,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        # detector
        super().__init__(name, cfg, input_tensor_size, device)
        self.imgsz = (input_tensor_size, input_tensor_size)
        self.stride, self.pt = None, None

    def load(self, model_weights, **args):
        w = str(model_weights[0] if isinstance(model_weights, list) else model_weights)
        self.detector = attempt_load(model_weights if isinstance(model_weights, list) else w,
                                     map_location=self.device, inplace=False)
        self.detector.eval()
        self.detector.requires_grad_(False)
        self.stride = max(int(self.detector.stride.max()), 32)  # model stride
        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names  # get class names

    # def init_img(self, img_numpy):
    #     # img_cv2 = cv2.resize(img, self.imgsz)
    #     img = np.transpose(img_numpy, (1, 2, 0))
    #     im = letterbox(img, self.imgsz, stride=self.stride, auto=True)[0]
    #     # Convert
    #     im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    #     im = np.ascontiguousarray(im)
    #
    #     im = torch.from_numpy(im).to(self.device).float()
    #     im /= 255  # 0 - 255 to 0.0 - 1.0
    #     if len(im.shape) == 3:
    #         im = im[None]  # expand for batch dim
    #
    #     img_tensor = Variable(im)  # , requires_grad=True
    #     return img_tensor
    #
    # def init_img_batch(self, img_numpy_batch):
    #     # Padded resize
    #     tensor_batch = None
    #     for img_numpy in img_numpy_batch:
    #         img_tensor = self.init_img(img_numpy)
    #         if tensor_batch is None:
    #             tensor_batch = img_tensor
    #         else:
    #             tensor_batch = torch.cat((tensor_batch, img_tensor), 0)
    #     # print(tensor_batch.shape)
    #     return tensor_batch.to(self.device)

    # def normalize(self, image_data):
    #     image_data = image_data.transpose((2, 0, 1))[::-1].astype(np.float)
    #     image_data = np.ascontiguousarray(image_data) / 255.
    #     img_tensor = torch.from_numpy(image_data).unsqueeze(0).float()
    #     img_tensor = Variable(img_tensor).to(self.device)
    #     img_tensor.requires_grad = True
    #     # print(img_tensor)
    #     return img_tensor, image_data
    #
    # def unnormalize(self, img_tensor):
    #     # img_tensor: tensor [1, c, h, w]
    #     img_numpy = img_tensor.squeeze(0).cpu().detach().numpy()
    #     img_numpy = np.transpose(img_numpy, (1, 2, 0))
    #     img_numpy *= 255
    #     img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
    #     img_numpy_int8 = img_numpy.astype('uint8')
    #     return img_numpy, img_numpy_int8

    def detect_img_batch_get_bbox_conf(self, batch_tensor, conf_thresh=0.5):
        # print("detect: ", batch_tensor.requires_grad)
        output = self.detector(batch_tensor, augment=False, visualize=False)[0]
        # print("output", output)
        preds = non_max_suppression(output.detach().cpu(),
                                    self.conf_thres, self.iou_thres) # [batch, num, 6] e.g., [1, 22743, 1, 4]
        bbox_array = []
        for pred in preds:
            # print(pred)
            box = scale_coords(batch_tensor.shape[-2:], pred, self.ori_size)
            box[:, [0,2]] /= self.ori_size[1]
            box[:, [1,3]] /= self.ori_size[0]
            bbox_array.append(box)

        # [batch, num, num_classes] e.g.,[1, 22743, 80]
        confs = output[..., 4]
        confs = confs[confs > conf_thresh]
        # print(bbox_array, confs)
        # print('filtered: ', confs.shape)
        return bbox_array, confs

