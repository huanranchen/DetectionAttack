import sys
import os
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import cv2

from .Pytorch_YOLOv4.tool.utils import *
from .Pytorch_YOLOv4.tool.torch_utils import *
from .Pytorch_YOLOv4.tool.darknet2pytorch import Darknet

from ...DetectorBase import DetectorBase


class HHYolov4(DetectorBase):
    def __init__(self, name, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        # name
        super().__init__(name, device)

    def load(self, model_weights, cfg_file=None, data_config_path=None):
        self.detector = Darknet(cfg_file)

        # data config
        self.num_classes = self.detector.num_classes
        if self.num_classes == 20:
            namesfile = os.path.join(data_config_path, 'data/voc.names')
        elif self.num_classes == 80:
            namesfile = os.path.join(data_config_path, 'data/coco.names')
        else:
            namesfile = os.path.join(data_config_path, 'data/x.names')
        self.namesfile = load_class_names(namesfile)

        # post_processing method | input (img, conf_thresh, nms_thresh, output) | return bboxes_batch
        self.post_processing = post_processing

        self.detector.load_weights(model_weights)
        self.detector.eval()
        self.detector.to(self.device)

        self.module_list = self.detector.models

    def init_img_batch(self, img_numpy_batch):
        img_tensor = Variable(torch.from_numpy(img_numpy_batch).float().div(255.0).to(self.device))
        return img_tensor

    def normalize(self, image_data):
        # init numpy into tensor & preprocessing (if mean-std(or other) normalization needed)
        # image_data: np array [h, w, c]
        # print('normalize: ', image_data.shape)
        image_data = np.array(image_data, dtype='float32') / 255.
        image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)
        img_tensor = torch.from_numpy(image_data).to(self.device)
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

    def detect_img_batch_get_bbox_conf(self, batch_tensor, conf_thresh=0.5, nms_thresh=0.4):
        output = self.detector(batch_tensor)
        # output: [ [batch, num, 1, 4], [batch, num, num_classes] ]
        confs = output[1]
        print(self.name, confs.shape)
        preds = self.post_processing(batch_tensor, conf_thresh, nms_thresh, output)
        for i, pred in enumerate(preds):
            preds[i] = np.array(pred) # shape([1, 6])
        # print('v4 h: ', confs.shape, confs.requires_grad)
        return preds, confs
