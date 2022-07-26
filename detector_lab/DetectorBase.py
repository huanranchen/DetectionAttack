import os
import sys
from abc import ABC, abstractmethod
import torch
import numpy as np
from torch.autograd import Variable
import cv2
from torchvision import transforms


class DetectorBase(ABC):
    def __init__(self,
                 name: str,
                 cfg: object,
                 input_tensor_size: int,
                 device: torch.device,
                 ):

        # detector title
        self.name = name
        # device (cuda or cpu)
        self.device = device

        # int(square:input_tensor_size * input_tensor_size): size of the tensor to input the detector
        self.input_tensor_size = input_tensor_size
        self.cfg = cfg

        self.conf_thres = cfg.CONF_THRESH
        self.iou_thres = cfg.NMS_THRESH
        self.ori_size = cfg.INPUT_SIZE

    def detach(self, tensor: torch.tensor):
        if self.device == torch.device('cpu'):
            return tensor.detach()
        return tensor.cpu().detach()

    def load_class_names(self, label_file):
        PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(PROJECT_DIR)
        from tools import load_class_names

        self.class_names = load_class_names(label_file)

    def zero_grad(self):
        self.detector.zero_grad()

    @abstractmethod
    def load(self, model_weights: str, **args):
        """
        init the detector model, load trained model weights and target detection classes file
        Args:
            input:
                model_weights
                **args: { classes_path }
            output:
        """
        # need to be loaded
        self.detector = None
        pass

    # @abstractmethod
    # def init_img_batch(self, img_numpy_batch: np.ndarray):
    #     """
    #     init img batch from numpy to tensor which can include custom preprocessing methods
    #     Args:
    #         input:
    #             img_numpy_batch: numpy image batch
    #         output:
    #             img_tensor: tensor batch
    #     """
    #     pass

    @abstractmethod
    def detect_img_batch_get_bbox_conf(self, batch_tensor: torch.tensor):
        """
        Detection core function, get detection results by feedding the input image
        Args:
            input:
                batch_tensor: image tensor [batch_size, channel, h, w]

            output:
                box_array: list of bboxes(batch_size*N*6) [[x1, y1, x2, y2, conf, cls_id],..]
                detections_with_grad: confidence of the object
        """
        pass

    def normalize(self, bgr_img_numpy: np.ndarray):
        '''
        normalize numpy data to tensor data
        Args:
            bgr_img_numpy: BGR numpy uint8 data HWC
        '''

        # convert to RGB
        image_data = cv2.cvtColor(bgr_img_numpy.astype(np.uint8), cv2.COLOR_BGR2RGB)

        # to CHW & normalize: auto scale when dtype=uint8
        image = transforms.ToTensor()(image_data)
        img_tensor = image.unsqueeze(0).to(self.device)
        # img_tensor.requires_grad = True
        return img_tensor

    def unnormalize(self, img_tensor: torch.tensor):
        '''
        unormalize the tensor into numpy
        Args:
            img_tensor: input tensor
        '''
        # img_tensor: tensor [1, c, h, w]
        img_numpy = self.detach(img_tensor.squeeze(0)).numpy()
        # convert to BGR & HWC
        img_numpy = img_numpy.transpose((1, 2, 0))
        img_numpy *= 255
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        img_numpy_int8 = img_numpy.astype('uint8')
        return img_numpy, img_numpy_int8


    def unnormalize_tensor(self, img_tensor: torch.tensor):
        """
        default: no unnormalization (depends on the detector)
        Rewrite the func if 'normalize_tensor' func has specifically edited
        """
        return img_tensor

    def normalize_tensor(self, img_tensor: torch.tensor, clone: bool =False):
        """
        normalize tensor (for patch)
        default: no normalization (depends on the detector), return a newly cloned tensor
        Rewrite the func if a normalization needed
        """
        tensor_data = img_tensor.clone()
        # print("normalize: ", tensor_data.is_leaf)
        return tensor_data

    def int8_precision_loss(self, img_tensor: torch.tensor):
        """
        (to stimulate the precision loss by dtype convertion in physical world)
        convert dtype of inout from float dtype to uint8 dtype, and convert back to the float dtype (including normalization)
        Args:
            input:
            img_tensor: detached torch tensor
        """
        img_tensor = self.unnormalize_tensor(img_tensor)
        img_tensor *= 255.
        img_tensor = img_tensor.to(torch.uint8)
        # print(img_tensor, torch.max(img_tensor), torch.min(img_tensor))
        img_tensor = img_tensor.to(torch.float)
        img_tensor /= 255.
        img_tensor = self.normalize_tensor(img_tensor)
        return img_tensor