import PIL.Image
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL.Image import Image


class FormatConverter:
    @staticmethod
    def PIL2tensor(PIL_image: Image):
        """

        :param PIL_image:
        :return: torch.tensor
        """
        return transforms.ToTensor()(PIL_image)

    @staticmethod
    def tensor2PIL(img_tensor: torch.tensor):
        """

        :param img_tensor: RGB image as torch.tensor
        :return: RGB PIL image
        """
        return transforms.ToPILImage()(img_tensor)

    @staticmethod
    def numpy2tensor(data):
        """

        :param data: rgb numpy
        :return: rgb torch tensor CBHW
        """
        return transforms.ToTensor()(data.astype(np.float)).unsqueeze(0) / 255.

    @staticmethod
    def bgr_numpy2tensor(bgr_img_numpy: np.ndarray):
        """

        :param bgr_img_numpy: BGR image in cv2 format
        :return: RGB image as torch.tensor
        """
        data = np.array(bgr_img_numpy, dtype='float32')
        rgb_im = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        img_tensor = FormatConverter.numpy2tensor(rgb_im)
        return img_tensor

    @staticmethod
    def tensor2numpy_cv2(im_tensor: torch.tensor):
        """

        :param im_tensor: RGB image in torch.tensor not in the computational graph & in cpu
        :return: BGR image in cv2 format
        """
        img_numpy = im_tensor.numpy().transpose((1, 2, 0))
        bgr_im = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR) * 255.
        bgr_im_int8 = bgr_im.astype('uint8')
        return bgr_im_int8