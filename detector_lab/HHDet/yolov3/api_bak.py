### 测试所用, 需要删除
import sys
from tkinter.tix import Tree

sys.path.append('/home/chenziyan/BaseDetectionAttack/detector_lab/PyTorch-YOLOv3/')
### 测试所用, 需要删除
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

import numpy as np
import cv2

# print(sys.path)


from pytorchyolo.models import load_model
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info, xywh2xyxy
from ...DetectorBase import DetectorBase


class HHYolov3(DetectorBase):
    def __init__(self, name, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        # name
        super().__init__(name, device)

    def load(self, model_weights, detector_config_file=None):
        """load model and weights

        Args:
            detector_config_file (str, optional): the config file of detector. Defaults to None.
            data_config_file (str, optional): the config file of dataset. Defaults to None.
            detector_weights (torch weights, optional): the torch weights of detector. Defaults to None.
        """
        self.detector = load_model(model_path=detector_config_file, weights_path=model_weights)
        self.detector.to(self.device)

    def init_img_batch(self, img_numpy_batch):
        tensor_batch = None
        for img in img_numpy_batch:
            img = np.transpose(img, (1, 2, 0))
            img_tensor = transforms.Compose([
                DEFAULT_TRANSFORMS])(
                (img, np.zeros((1, 5))))[0].unsqueeze(0)
            if tensor_batch is None:
                tensor_batch = img_tensor
            else:
                tensor_batch = torch.cat((tensor_batch, img_tensor), 0)

        return tensor_batch.to(self.device)

    def prepare_img(self, img_path=None, img_cv2=None, input_shape=(416, 416)):
        """prepare a image from img path or cv2 img

        Args:
            img_path (str, optional): the path of input image. Defaults to None.
            img_cv2 (np.numpy, optional): the cv2 type image. Defaults to None.
            input_shape (int, optional): the size to resize. Defaults to 416.

        Raises:
            Exception: if no input image, raise the exception

        Returns:
            tuple: [torch.Tensor, np.numpy], the torch tensor of input image, the cv2 type image of input
        """

        if img_path:
            img_cv2 = cv2.imread(img_path)
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        elif img_cv2:
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        else:
            raise Exception('no input image!')

        # print('cv2', np.max(img_cv2))
        input_img = transforms.Compose([
            DEFAULT_TRANSFORMS,
            Resize(input_shape[0])])(
            (img_cv2, np.zeros((1, 5))))[0].unsqueeze(0)
        # print('max: ', input_img[0].max())
        input_img = input_img.to(self.device)
        # input_img.requires_grad = True
        return input_img, img_cv2

    def normalize(self, image_data):
        # image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_data = np.array(image_data, dtype='float32')
        # print('normalize', image_data.shape)
        image_data /= 255.0
        image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)
        img_tensor = torch.from_numpy(image_data).to(self.device)
        img_tensor.requires_grad = True
        # print('normalize: ', img_tensor.is_leaf)
        return img_tensor, image_data

    # def normalize_tensor(self, img_tensor):
    #     # for if mean-std(or other) normalization needed
    #     tensor_data = img_tensor.clone()
    #     tensor_data.requires_grad = True
    #     return tensor_data

    def unnormalize(self, img_tensor):
        # img_tensor: tensor [1, c, h, w]
        img_numpy = img_tensor.squeeze(0).cpu().detach().numpy()
        img_numpy = np.transpose(img_numpy, (1, 2, 0))
        img_numpy *= 255
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        img_numpy_int8 = img_numpy.astype('uint8')
        return img_numpy, img_numpy_int8

    # def unnormalize_tensor(self, img_tensor):
    #     return img_tensor

    def detect_img_batch_get_bbox_conf(self, batch_tensor, input_size=416, conf_thres=0.5, nms_thres=0.4):
        # output: box_array: list of np N*6
        self.detector.eval()
        ori_shape = batch_tensor.shape[-2:]
        detections_with_grad = self.detector(batch_tensor)
        preds = non_max_suppression(detections_with_grad, conf_thres, nms_thres)

        bbox_array = []
        for pred in preds:
            box = rescale_boxes(pred, input_size, ori_shape)
            box[:, 0] /= ori_shape[1]
            box[:, 1] /= ori_shape[0]
            box[:, 2] /= ori_shape[1]
            box[:, 3] /= ori_shape[0]
            bbox_array.append(np.array(box))
        # print(bbox_array)
        return bbox_array, detections_with_grad[:, :, 4]

    def detect_img_tensor_get_bbox_conf(self, input_img, ori_img_cv2, img_size=416, conf_thres=0.5, nms_thres=0.4):
        self.detector.eval()
        detections_with_grad = self.detector(input_img)
        detections = non_max_suppression(detections_with_grad, conf_thres, nms_thres)
        detections = rescale_boxes(detections[0], img_size, ori_img_cv2.shape[:2])
        detections[:, 0] /= ori_img_cv2.shape[1]
        detections[:, 1] /= ori_img_cv2.shape[0]
        detections[:, 2] /= ori_img_cv2.shape[1]
        detections[:, 3] /= ori_img_cv2.shape[0]
        # print(detections.shape)
        # detections = detections.numpy().tolist()
        # print(np.array(detections).shape)

        return detections, detections_with_grad[:, :, 4]

    def temp_loss(self, confs):
        return torch.nn.MSELoss()(confs.to(self.device), torch.ones(confs.shape).to(self.device))


if __name__ == '__main__':
    hhyolov3 = HHYolov3(name="YOLOV3")
    hhyolov3.load(detector_config_file='./detector_lab/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3.cfg',
                  model_weights='./detector_lab/HHDet/yolov3/PyTorch_YOLOv3/weights/yolov3.weights')

    input_img, img_cv2 = hhyolov3.prepare_img(
        '/home/huanghao/BaseDetectionAttack/detector_lab/pytorch-YOLOv4/data/giraffe.jpg')
    box_array, confs = hhyolov3.detect_img_tensor_get_bbox_conf(input_img, img_cv2)
    loss = hhyolov3.temp_loss(confs)
    print(loss.item())
    loss.backward()
    print(input_img.requires_grad)
    print(input_img.grad)

    # 注意 这里面用的一个包 pytorchyolo
    #  File "/home/huanghao/anaconda3/envs/dassl/lib/python3.7/site-packages/pytorchyolo/models.py", line 163, in _make_grid
    # yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
    # 会有这个问题
    # 需要改成 yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])