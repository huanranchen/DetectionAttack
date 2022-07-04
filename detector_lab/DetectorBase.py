from abc import ABC, abstractmethod
import torch


class DetectorBase(ABC):
    def __init__(self, name, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        # detector title
        self.name = name
        # device (cuda or cpu)
        self.device = device
        # need to be loaded by calling self.load()
        self.detector = None

    def detach(self, tensor):
        if self.device == torch.device('cpu'):
            return tensor.detach()
        return tensor.cpu().detach()

    def zero_grad(self):
        self.detector.zero_grad()

    @abstractmethod
    def load(self, model_weights, **args):
        """
        init the detector model, load trained model weights and target detection classes file
        Args:
            input:
                model_weights
                classes_path
            output:
        """
        pass

    @abstractmethod
    def init_img_batch(self, img_numpy_batch):
        """
        init img batch from numpy to tensor which can include customed preprocessing methods
        Args:
            input:
                img_numpy_batch: numpy image batch
            output:
                img_tensor: tensor batch
        """
        pass

    @abstractmethod
    def detect_img_batch_get_bbox_conf(self, batch_tensor):
        """
        Detection core function, get detection results by feedding the input image
        Args:
            input:
                batch_tensor: image tensor [batch_size, channel, h, w]

            output:
                box_array: list of bboxes(batch_size*N*6) [[x1,y1,x2,y2,cls,cls_conf],..]
                detections_with_grad: confidence of the object
        """
        pass

    def unnormalize_tensor(self, img_tensor):
        """
        default: no unnormalization (depends on the detector)
        """
        return img_tensor

    def normalize_tensor(self, img_tensor):
        """
        default: no normalization (depends on the detector)
        Rewrite if normalization needed
        """
        tensor_data = img_tensor.clone()
        tensor_data.requires_grad = True
        return tensor_data

    def int8_precision_loss(self, img_tensor):
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