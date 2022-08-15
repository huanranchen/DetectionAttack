import torch
import numpy as np

from .PyTorch_YOLOv3.pytorchyolo.models import load_model
from .PyTorch_YOLOv3.pytorchyolo.utils.loss import build_targets
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

    def requires_grad_(self, state):
        self.detector.module_list.requires_grad_(state)
    
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
        if not hasattr(self.cfg, 'GRAD_PERTURB') or not self.cfg.GRAD_PERTURB:
            print(self.name, ' eval & requires no grad')
            self.requires_grad_(False)
            self.detector.eval()
        else:
            self.gradient_opt()


    def detect_img_batch_get_bbox_conf(self, batch_tensor, confs_thresh=0.5):
        detections_with_grad = self.detector(batch_tensor) # torch.tensor([1, num, classes_num+4+1])
        # if self.detector.training:
        #     print(torch.cat(detections_with_grad, 1).shape)
        # print(len(detections_with_grad))
        preds = non_max_suppression(detections_with_grad, self.conf_thres, self.iou_thres)
        confs = detections_with_grad[:, :, 4]
        # index = confs > 0.5
        # print(len(preds))
        bbox_array = []
        for i, pred in enumerate(preds):
            box = np.array(rescale_boxes(pred, self.input_tensor_size, self.ori_size))
            # print(box)
            box[:, [0, 2]] /= self.ori_size[1]
            box[:, [1, 3]] /= self.ori_size[0]
            box[:, :4] = np.clip(box[:, :4], a_min=0, a_max=1)
            # print(box)
            bbox_array.append(box)

            # # TODO: CONF_POLICY test
            # if hasattr(self.cfg, 'CONF_POLICY') and self.cfg.CONF_POLICY and self.target is not None:
            #     # print('CONF_POLICY: photo ', i, '/', len(self.target))
            #     # print('Currently bbox num: ', len(box), '/', len(self.target[i]))
            # ind = confs[i] > confs_thresh
            #     # print(confs[i][ind])
            #
            #     if len(box) == 0 or i >= len(self.target):
            #         confs[i][ind] = torch.zeros(1).to(self.device)
            #     else:
            #         confs[i][ind] *= (len(box)/len(self.target[i]))
                # print(confs[i][ind])
                # print('----------------')
        confs = confs[confs > confs_thresh]
        return bbox_array, confs