import colorsys
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import cv2
from PIL import ImageDraw, ImageFont

from PIL import Image
from .yolox.nets.yolo import YoloBody
from .yolox.utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from .yolox.utils.utils_bbox import decode_outputs, non_max_suppression
from ...DetectorBase import DetectorBase

# from ..utils import BaseDetector

class YOLO(DetectorBase):
    _defaults = {
        "model_path"        : 'yolox/logs/ep1155-loss1.474-val_loss1.452.pth',
        "classes_path"      : 'yolox/model_data/car_classes.txt',
        "input_shape"       : [416, 416],
        "phi"               : 'x',
        "confidence"        : 0.5,
        "nms_iou"           : 0.3,
        "letterbox_image"   : False,
        "cuda"              : True,
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), **kwargs):
        super().__init__()
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        
        self.device = device
        self.detector = None
        self.name = name
        # self.generate()

    # def zero_grad(self):
    #     self.detector.zero_grad()

    def temp_loss(self, confs):
        return torch.nn.MSELoss()(confs.to(self.device), torch.ones(confs.shape).to(self.device))

    def load(self, model_weights, classes_path):
        self.class_names, self.num_classes  = get_classes(classes_path)
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.detector    = YoloBody(self.num_classes, self.phi)
        self.detector.load_state_dict(torch.load(model_weights, map_location=self.device))
        self.detector    = self.detector.eval()

        print('{} model, and classes loaded.'.format(model_weights))

        self.detector = self.detector.to(self.device)

    def prepare_img(self, img_path=None, img_cv2=None, input_shape=None):
        if img_path:
            image = Image.open(img_path)
        else:
            image = Image.fromarray(cv2.cvtColor(img_cv2.astype('uint8'), cv2.COLOR_BGRA2RGBA))
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        img_tensor = torch.from_numpy(image_data).to(self.device)
        # img_tensor.requires_grad = True

        return img_tensor, np.array(image)

    def init_img_batch(self, img_numpy_batch):
        tensor_batch = None
        for img in img_numpy_batch:
            img = np.transpose(np.array(img, dtype='float32'), (1, 2, 0))
            img = np.expand_dims(
                np.transpose(preprocess_input(np.array(img, dtype='float32')), (2, 0, 1)), 0)
            img_tensor = torch.from_numpy(img)
            if tensor_batch is None:
                tensor_batch = img_tensor
            else:
                tensor_batch = torch.cat((tensor_batch, img_tensor), 0)
        return tensor_batch.to(self.device)
    
    def normalize(self, image_data):
        # input numpy img, return normalized tensor
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        img_tensor = torch.from_numpy(image_data).to(self.device)
        img_tensor.requires_grad = True
        # print(img_tensor.is_leaf)
        return img_tensor, np.array(image_data)

    def unnormalize(self, img_tensor):
        # img_tensor: tensor [b, c, h, w] (RGB channel)
        img_numpy = img_tensor.squeeze(0).cpu().detach().numpy()
        img_numpy = np.transpose(img_numpy, (1, 2, 0))
        img_numpy *= np.array([0.229, 0.224, 0.225])
        img_numpy += np.array([0.485, 0.456, 0.406])
        img_numpy *= 255

        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        img_numpy_int8 = img_numpy.astype('uint8')

        return img_numpy, img_numpy_int8

    def normalize_tensor(self, img_tensor):
        # input tensor img, return normalized tensor
        tensor_data = img_tensor.clone()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # print('normalize', tensor_data.is_leaf, tensor_data.requires_grad)
        tensor_data[0, 0, ...] = (tensor_data[0, 0, ...] - mean[0]) / std[0]
        tensor_data[0, 1, ...] = (tensor_data[0, 1, ...] - mean[1]) / std[1]
        tensor_data[0, 2, ...] = (tensor_data[0, 2, ...] - mean[2]) / std[2]
        tensor_data.requires_grad = True
        # print('r channel',tensor_data[0, 0,...].max())
        # print('g channel', tensor_data[0, 1, ...].max())
        # print('b channel', tensor_data[0, 2, ...].max())
        # print('normalize2', tensor_data.is_leaf, tensor_data.requires_grad)
        return tensor_data

    def unnormalize_tensor(self, tensor_data):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tensor_data[0, 0, ...] = tensor_data[0, 0, ...] * std[0] + mean[0]
        tensor_data[0, 1, ...] = tensor_data[0, 1, ...] * std[1] + mean[1]
        tensor_data[0, 2, ...] = tensor_data[0, 2, ...] * std[2] + mean[2]
        return tensor_data


    def detect(self, img_tensor):
        outputs = self.detector(img_tensor)
        results_with_grad = decode_outputs(outputs, self.input_shape)
        return results_with_grad

    def box_rectify(self, result, image_shape):
        box = np.array([])
        if result.shape[0] > 0:
            top_label = np.array(result[:, 6], dtype='int32')
            confs = result[:, 4] * result[:, 5]
            top_boxes = result[:, :4]
            # rescale bbox to [0, 1]
            top_boxes[:, [0, 2]] = (top_boxes[:, [0, 2]] / image_shape[0]).clip(0, 1)
            top_boxes[:, [1, 3]] = (top_boxes[:, [1, 3]] / image_shape[1]).clip(0, 1)
            top_boxes_new = copy.deepcopy(top_boxes)
            top_boxes_new[:, [0, 2]] = top_boxes[:, [1, 3]]
            top_boxes_new[:, [1, 3]] = top_boxes[:, [0, 2]]
            box = np.c_[top_boxes_new, confs, top_label]
        return box

    def detect_img_batch_get_bbox_conf(self, batch_tensor):
        # print(batch_tensor.shape)
        results_with_grad = self.detect(batch_tensor)
        results = results_with_grad.clone().detach()
        results = non_max_suppression(results, self.num_classes, self.input_shape,
                                      self.input_shape, self.letterbox_image, conf_thres=self.confidence,
                                      nms_thres=self.nms_iou)
        # print('batch: ', len(results))
        box_array = []
        for result in results:
            box = self.box_rectify(result, self.input_shape)
            box_array.append(box)
        # print('yolox', box_array)
        # box_array: [batch_size, bbox_num, [xyxy, confs, cls_label]]
        # results_with_grad: [batch_size, raw_obj_confs]
        return box_array, results_with_grad[:, :, 4]

    def detect_img_tensor_get_bbox_conf(self, input_img, ori_img_cv2):
        image_shape = np.array(np.shape(ori_img_cv2)[0:2])
        results_with_grad = self.detect(input_img)

        # 1*x*9: [b1, b2, b3, b4, p_obj, p_class...(num_classes)]
        results = results_with_grad.clone().detach()
        print(results)
        results = non_max_suppression(results, self.num_classes, self.input_shape,
                                      image_shape, self.letterbox_image, conf_thres=self.confidence,
                                      nms_thres=self.nms_iou)
        # print('results_with_grad: ', results_with_grad[0, 1, :], results_with_grad.shape)
        # print('results:', results[0].shape)
        box_array = self.box_rectify(results[0], image_shape)
        return box_array, results_with_grad[0, :, 4]


if __name__ == "__main__":
    import os
    from tqdm import tqdm

    dir_origin_path = r"yolox/img/"
    dir_save_path = r"yolox/img_out/"
    img_names = os.listdir(dir_origin_path)

    yolo = YOLO()
    yolo.load()
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_origin_path, img_name)
            img_tensor, image = yolo.prepare_img(image_path)
            box_array, confs_with_grad = yolo.detect_img_tensor_get_bbox_conf(img_tensor, image)
            print('box', box_array, confs_with_grad.shape)
            loss = yolo.temp_loss(confs_with_grad)
            # print(loss.item())
            loss.backward()
            # print(img_tensor.requires_grad)
            # print(img_tensor.grad)

            # show detection results with boxes
            yolo.detect_cv2_show(image_path, dir_save_path, img_name)
