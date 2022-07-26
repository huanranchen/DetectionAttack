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


class HHYolov5:
    def __init__(self, name, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.name = name
        self.device = device
        self.detector = None
        self.imgsz = (640, 640)
        self.detector, self.stride, self.names, self.pt = None, None, None, None

    def zero_grad(self):
        self.detector.zero_grad()

    def load(self, model_weights, **args):
        w = str(model_weights[0] if isinstance(model_weights, list) else model_weights)
        self.detector = attempt_load(model_weights if isinstance(model_weights, list) else w, map_location=self.device,
                                     inplace=False)
        self.stride = max(int(self.detector.stride.max()), 32)  # model stride
        self.names = self.detector.module.names if hasattr(self.detector,
                                                           'module') else self.detector.names  # get class names

    def prepare_img(self, img_path=None, img_cv2=None, input_shape=(640, 640)):
        """prepare a image from img path or cv2 img

        Args:
            img_path (str, optional): the path of input image. Defaults to None.
            img_cv2 (np.numpy, optional): the cv2 type image. Defaults to None.
            img_size (tuple, optional): the size to resize. Defaults to 416.

        Raises:
            Exception: if no input image, raise the exception

        Returns:
            tuple: [torch.Tensor, np.numpy], the torch tensor of input image, the cv2 type image of input
        """

        if img_path:
            img_cv2 = cv2.imread(img_path)
        elif img_cv2:
            img_cv2 = img_cv2
        else:
            raise Exception('no input image!')

        # Padded resize
        img_cv2 = cv2.resize(img_cv2, input_shape)
        im = letterbox(img_cv2, self.imgsz, stride=self.stride, auto=True)[0]
        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device).float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        img_tensor = Variable(im)  # , requires_grad=True
        return img_tensor, img_cv2

    def normalize(self, image_data):
        image_data = image_data.squeeze(0).transpose((2, 0, 1))[::-1]
        image_data = np.ascontiguousarray(image_data) / 255.
        img_tensor = torch.from_numpy(im).to(self.device).float()
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

    def detect_cv2_show(self, imgfile, savename='predictions.jpg'):
        """detect a image with yolov5 and draw the bounding boxes

        Args:
            imgfile ([str]): [the path of image to be detected]

        Returns:
            boxes[0] ([list]): [detected boxes]
        """
        im0s = cv2.imread(imgfile)
        # Padded resize
        im = letterbox(im0s, self.imgsz, stride=self.stride, auto=True)[0]
        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)

        im = torch.from_numpy(im).to(self.device).float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.detector(im, augment=False, visualize=False)[0]  # pred: torch.Size([1, 18900, 85])
        # NMS, pred is a list, [torch.Size([5, 6])], [x, y, x, y, conf, class_idx], [x, y, w, h] Corresponding to imgsz
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, max_det=1000)
        det = pred[0]  # for signle
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
        self.plot_boxes_cv2(imgfile, det, savename)

    def plot_boxes_cv2(self, imgfile, boxes, savename=None):
        """[summary]

        Args:
            imgfile ([cv2.image]): [the path of image to be detected]
            boxes ([type]): [detected boxes]
            savename ([str], optional): [save image name]. Defaults to None.

        Returns:
            [cv2.image]: [cv2 type image with drawn boxes]
        """
        im0 = cv2.imread(imgfile)
        annotator = Annotator(im0, line_width=3, example=str(self.names))
        if len(boxes):
            for *xyxy, conf, cls in reversed(boxes):
                c = int(cls)  # integer class
                label = self.names[c] if False else f'{self.names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

        if savename:
            print("save plot results to %s" % savename)
            cv2.imwrite(savename, im0)

    def detect_img_tensor_get_bbox_conf(self, img_tensor):
        # self.detector.eval()
        output = self.detector(img_tensor, augment=False, visualize=False)[0]
        # [batch, num, 1, 4] e.g., [1, 22743, 1, 4]
        box_array = output[:, :, :4].unsqueeze(2)
        # [batch, num, num_classes] e.g.,[1, 22743, 80]
        confs = output[:, :, 5:]
        return box_array, confs

    def temp_loss(self, confs):
        return torch.nn.MSELoss()(confs, torch.ones(confs.shape).to(self.device))


if __name__ == '__main__':
    img_path = '/home/chenziyan/work/BaseDetectionAttack/data/coco/train/train2017/000000000030.jpg'
    sys.path.append('~/work/BaseDetectionAttack/detector_lab/HHDet/yolov5')
    hhyolov5 = HHYolov5(name="YOLOV5")
    hhyolov5.load(model_weights='./detector_lab/HHDet/yolov5/yolov5/weight/yolov5s.pt')
    hhyolov5.detect_cv2_show(img_path)

    im0s = cv2.imread(img_path)
    # Padded resize
    im = letterbox(im0s, hhyolov5.imgsz, stride=hhyolov5.stride, auto=True)[0]
    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(hhyolov5.device).float()
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    img_tensor = Variable(im, requires_grad=True)
    print(img_tensor.shape)
    box_array, confs = hhyolov5.detect_img_tensor_get_bbox_conf(img_tensor)
    loss = hhyolov5.temp_loss(confs)
    loss.backward()
    print(img_tensor.grad, img_tensor.grad.size())
