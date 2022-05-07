import colorsys
from torch.nn import functional as F
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import Image, ImageDraw, ImageFont

from .faster_rcnn.nets.frcnn import FasterRCNN
from .faster_rcnn.utils.utils import (cvtColor, get_classes, get_new_img_size, resize_image,
                         preprocess_input)
from .faster_rcnn.utils.utils_bbox import DecodeBox
import os
import copy

class FRCNN(object):
    _defaults = {
        "model_path"    : 'faster_rcnn/logs/ep300-loss0.701-val_loss1.057.pth',
        "classes_path"  : 'faster_rcnn/model_data/car_classes.txt',
        "backbone"      : "resnet50",
        "confidence"    : 0.5,
        "nms_iou"       : 0.3,
        'anchors_size'  : [8, 16, 32],
        "cuda"          : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        self.device = device
        # self.generate()
        self.detector = None
        self.name = name

    def zero_grad(self):
        self.detector.zero_grad()

    def load(self, detector_weights, classes_path):
        self.class_names, self.num_classes  = get_classes(classes_path)
        self.std    = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(self.num_classes + 1)[None]
        self.std    = self.std.to(self.device)
        self.bbox_util  = DecodeBox(self.std, self.num_classes)


        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        
        self.detector    = FasterRCNN(self.num_classes, "predict", anchor_scales = self.anchors_size, backbone = self.backbone)
        self.detector.load_state_dict(torch.load(detector_weights, map_location=self.device))
        self.detector    = self.detector.eval()
        print('{} model, anchors, and classes loaded.'.format(detector_weights))
        
        self.detector = self.detector.to(self.device)

    def init_img_batch(self, img_numpy_batch):

        img_tensor = torch.from_numpy(img_numpy_batch).float().div(255.0).to(self.device)
        return img_tensor

    def prepare_img(self, img_path=None, img_cv2=None, input_shape=(416, 416)):
        if img_path:
            image = Image.open(img_path)
        else:
            image = Image.fromarray(cv2.cvtColor(img_cv2.astype('uint8'), cv2.COLOR_BGRA2RGBA))
        size = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, [input_shape[1], input_shape[0]])
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        img_tensor = torch.from_numpy(image_data).to(self.device)
        #print(np.array(image).shape, 233333)
        # img_tensor.requires_grad = True
        return img_tensor, np.array(image)
    
    def normalize(self, image_data):
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        img_tensor = torch.from_numpy(image_data).to(self.device)
        img_tensor.requires_grad = True
        return img_tensor, np.array(image_data)
    
    def unnormalize(self, img_tensor):
        img_numpy = img_tensor.squeeze(0).cpu().detach().numpy() * 255.
        img_numpy = np.transpose(img_numpy, (1, 2, 0))
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        img_numpy_int8 = img_numpy.astype('uint8')
        return img_numpy, img_numpy_int8

    def normalize_tensor(self, img_tensor):
        # input tensor has a value region of [0, 1]
        tensor_data = img_tensor.clone()
        tensor_data.requires_grad = True
        return tensor_data

    def unnormalize_tensor(self, img_tensor):
        return img_tensor

    def temp_loss(self, confs):
        return torch.nn.MSELoss()(confs.to(self.device), torch.ones(confs.shape).to(self.device))

    def detect(self, img_tensor):
        roi_cls_locs, roi_scores, rois, _, rpn_scores = self.detector(img_tensor)
        return roi_cls_locs, roi_scores, rois, rpn_scores

    def box_rectify(self, result, image_shape):
        box = np.array([])
        if len(result) > 0:
            result[:, [0, 2]] = (result[:, [0, 2]] / image_shape[0]).clip(0, 1)
            result[:, [1, 3]] = (result[:, [1, 3]] / image_shape[1]).clip(0, 1)
            box = copy.deepcopy(result)
            box[:, [0, 2]] = result[:, [1, 3]]
            box[:, [1, 3]] = result[:, [0, 2]]
        return box

    def detect_img_batch_get_bbox_conf(self, input_img):
        image_shape = input_img.shape[-2:]
        roi_cls_locs, roi_scores, rois, rpn_scores = self.detect(input_img)
        # print('rpn:', rpn_scores[0, 0, :], rpn_scores.shape)
        results = self.bbox_util.forward(roi_cls_locs.clone().detach(), roi_scores.clone().detach(),
                                         rois.clone().detach(), image_shape, image_shape,
                                         nms_iou=self.nms_iou, confidence=self.confidence)
        # print('batch:', results, len(results))
        box_array = []
        for result in results:
            box = self.box_rectify(result, image_shape)
            box_array.append(box)
        # print('frcnn', box_array)
        return box_array, rpn_scores[:, :, 0]

    def detect_img_tensor_get_bbox_conf(self, input_img, ori_img_cv2):
        image_shape = np.array(np.shape(ori_img_cv2)[0:2])
        # input_shape = get_new_img_size(image_shape[0], image_shape[1])
        input_shape = image_shape[1], image_shape[0]

        roi_cls_locs, roi_scores, rois, rpn_scores = self.detect(input_img)
        # print('rpn:', rpn_scores[0, 0, :], rpn_scores.shape)
        results = self.bbox_util.forward(roi_cls_locs.clone().detach(), roi_scores.clone().detach(),
                                         rois.clone().detach(), image_shape, input_shape,
                                         nms_iou=self.nms_iou, confidence=self.confidence)
        # print('batch:', results, len(results))
        results = self.box_rectify(results[0], image_shape)
        return results, rpn_scores[:, :, 0]

class FRCNNAgent(FRCNN):
    def __init__(self, input_size, name, **kwargs):
        super().__init__(name, **kwargs)
        self.input_size = input_size

    def detect_img_batch_get_bbox_conf(self, img_batch):
        roi_cls_locs, roi_scores, rois, rpn_scores = self.detect(img_batch)
        results = self.bbox_util.forward(roi_cls_locs.clone().detach(), roi_scores.clone().detach(),
                                         rois.clone().detach(), self.input_size, self.input_size,
                                         nms_iou=self.nms_iou, confidence=self.confidence)
        box_array = []
        for result in results:
            box = self.box_rectify(result, self.input_size)
            box_array.append(box)
        return box_array, rpn_scores[:,:,0]


if __name__ == "__main__":
    from tqdm import tqdm

    dir_origin_path = "faster_rcnn/img/"
    dir_save_path = "faster_rcnn/img_out/"
    img_names = os.listdir(dir_origin_path)

    detector = FRCNN()
    detector.load()
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path  = os.path.join(dir_origin_path, img_name)
            img_tensor, image = detector.prepare_img(image_path)
            box_array, confs = detector.detect_img_tensor_get_bbox_conf(img_tensor, image)
            print('output:', np.array(box_array), confs.shape)
            loss = detector.temp_loss(confs)
            # print(loss.item())
            loss.backward()
            # print(img_tensor.requires_grad)
            # print(img_tensor.grad, confs.grad)

            # draw results
            detector.detect_cv2_show(image_path, dir_save_path, img_name)
            # break