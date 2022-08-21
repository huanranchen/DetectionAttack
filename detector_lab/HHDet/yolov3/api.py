import torch
import numpy as np

from .PyTorch_YOLOv3.pytorchyolo.models import load_model
from .PyTorch_YOLOv3.pytorchyolo.utils.utils import rescale_boxes, non_max_suppression
from ...DetectorBase import DetectorBase


class HHYolov3(DetectorBase):
    def __init__(self,
                 name, cfg, input_tensor_size=412,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.target = None

    def requires_grad_(self, state: bool):
        self.detector.module_list.requires_grad_(state)
    
    def load(self, model_weights, detector_config_file=None):
        if self.cfg.PERTURB.GATE == 'shake_drop':
            tmp = detector_config_file.split('/')
            detector_cfg = self.cfg.PERTURB.SHAKE_DROP.MODEL_CONFIG
            print('Self ensemble! Shake drop model cfg :', detector_config_file)
            tmp[-1] = detector_cfg
            self.detector = load_model(model_path='/'.join(tmp), weights_path=model_weights).to(self.device)
            self.clean_model = load_model(model_path=detector_config_file, weights_path=model_weights).to(self.device)
            self.clean_model.eval()
            self.clean_model.module_list.requires_grad_(False)
        else:
            self.detector = load_model(model_path=detector_config_file, weights_path=model_weights).to(self.device)
        self.eval()

        if self.cfg.PERTURB.GATE == 'grad_descend':
            print('Self ensemble: ', self.name, 'Grad descend...')
            self.gradient_opt()

    def detect_img_batch_get_bbox_conf(self, batch_tensor, confs_thresh=0.5, clean_model=False):
        if clean_model and hasattr(self, 'clean_model'):
            detections_with_grad = self.clean_model(batch_tensor)
        else:
            detections_with_grad = self.detector(batch_tensor) # torch.tensor([1, num, classes_num+4+1])
        preds = non_max_suppression(detections_with_grad, self.conf_thres, self.iou_thres)
        confs = detections_with_grad[:, :, 4]

        bbox_array = []
        for i, pred in enumerate(preds):
            box = np.array(rescale_boxes(pred, self.input_tensor_size, self.ori_size))
            # print(box)
            box[:, [0, 2]] /= self.ori_size[1]
            box[:, [1, 3]] /= self.ori_size[0]
            box[:, :4] = np.clip(box[:, :4], a_min=0, a_max=1)
            # print(box)
            bbox_array.append(box)

        confs = confs[confs > confs_thresh]
        return bbox_array, confs