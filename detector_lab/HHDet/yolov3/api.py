import torch
import numpy as np

from .PyTorch_YOLOv3.pytorchyolo.models import load_model
from .PyTorch_YOLOv3.pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from .PyTorch_YOLOv3.pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info, xywh2xyxy
from ...DetectorBase import DetectorBase


class HHYolov3(DetectorBase):
    def __init__(self,
                 name, cfg,
                 input_tensor_size=412,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        # name
        super().__init__(name, cfg, input_tensor_size, device)
        self.target = None

    
    def load(self, model_weights, detector_config_file=None):
        """load model and weights

        Args:
            detector_config_file (str, optional): the config file of detector. Defaults to None.
            data_config_file (str, optional): the config file of dataset. Defaults to None.
            detector_weights (torch weights, optional): the torch weights of detector. Defaults to None.
        """
        self.detector = load_model(model_path=detector_config_file, weights_path=model_weights)
        self.module_list = self.detector.module_list
        self.detector.to(self.device)
        self.detector.eval()
        self.detector.module_list.requires_grad_(False)

    # def init_img(self, img_numpy):
    #     # img_numpy: uint8, [0, 255], RGB, CHW
    #     img = np.transpose(img_numpy, (1, 2, 0))
    #     img_tensor = transforms.Compose([
    #         DEFAULT_TRANSFORMS])(
    #         (img, np.zeros((1, 5))))[0].unsqueeze(0)
    #     return img_tensor
    #
    # def init_img_batch(self, img_numpy_batch):
    #     tensor_batch = None
    #     for img in img_numpy_batch:
    #         img_tensor = self.init_img(img)
    #         if tensor_batch is None:
    #             tensor_batch = img_tensor
    #         else:
    #             tensor_batch = torch.cat((tensor_batch, img_tensor), 0)
    #     return tensor_batch.to(self.device)

    def detect_img_batch_get_bbox_conf(self, batch_tensor, conf_thresh=0.5):
        detections_with_grad = self.detector(batch_tensor) # torch.tensor([1, num, classes_num+4+1])
        preds = non_max_suppression(detections_with_grad, self.conf_thres, self.iou_thres)
        confs = detections_with_grad[:, :, 4]
        # index = confs > 0.5
        # print(len(preds))
        bbox_array = []
        for i, pred in enumerate(preds):
            box = rescale_boxes(pred, self.input_tensor_size, self.ori_size)
            # print(box)
            box[:,0] /= self.ori_size[1]
            box[:,1] /= self.ori_size[0]
            box[:,2] /= self.ori_size[1]
            box[:,3] /= self.ori_size[0]
            # print(box)
            bbox_array.append(np.array(box))

            # if self.target is not None:
                # print(len(box), len(self.target[i]))
                # ind = confs[i] > 0.5
                # print(confs[i][ind].shape)
                # print(i, len(self.target))
                # if len(box) == 0 or i >= len(self.target):
                #     confs[i][ind] = 0
                # else:
                #     confs[i][ind] *= (len(box)/len(self.target[i]))
        confs = confs[confs > conf_thresh]
        return bbox_array, confs