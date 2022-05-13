from abc import ABC, abstractmethod
import torch


class DetectorBase(ABC):
    def __init__(self):
        self.detector = None

    def zero_grad(self):
        self.detector.zero_grad()

    @abstractmethod
    def load(self, detector_weights, classes_path):
        '''
        init the detector model, load trained model weights and target detection classes file
        Args:
            input:
            output:
        '''
        pass

    @abstractmethod
    def init_img_batch(self, img_numpy_batch):
        '''
        init img batch from numpy to tensor which can include customed preprocessing methods
        Args:
            input:
            img_numpy_batch:

            output:
        '''
        pass

    @abstractmethod
    def detect_img_batch_get_bbox_conf(self, input_img):
        '''
        Detection core function, get detection results by feedding the input image
        Args:
            input:
            input_img

            output:
            box_array: list of bboxes(batch_size*N*6)
            detections_with_grad:
        '''
        pass

    def unnormalize_tensor(self, img_tensor):
        return img_tensor

    def normalize_tensor(self, img_tensor):
        return img_tensor

    def int8_precision_loss(self, img_tensor):
        '''
        (to stimulate the precision loss by dtype convertion in physical world)
        convert dtype of inout from float dtype to uint8 dtype, and convert back to the float dtype (including normalization)
        Args:
            input:
            img_tensor: detached torch tensor
        '''
        img_tensor = self.unnormalize_tensor(img_tensor)
        img_tensor *= 255.
        img_tensor = img_tensor.to(torch.uint8)
        # print(img_tensor, torch.max(img_tensor), torch.min(img_tensor))
        img_tensor = img_tensor.to(torch.float)
        img_tensor /= 255.
        img_tensor = self.normalize_tensor(img_tensor)
        return img_tensor