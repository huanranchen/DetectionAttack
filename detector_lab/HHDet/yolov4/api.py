import numpy as np

from .Pytorch_YOLOv4.tool.utils import *
from .Pytorch_YOLOv4.tool.torch_utils import *
from .Pytorch_YOLOv4.tool.darknet2pytorch import Darknet

from ...DetectorBase import DetectorBase


class HHYolov4(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_size=416,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)

    def requires_grad_(self, state: bool):
        assert self.detector
        self.detector.models.requires_grad_(state)

    def load(self, model_weights, detector_config_file=None, data_config_path=None):
        if self.cfg.PERTURB.GATE == 'shake_drop':
            from .Pytorch_YOLOv4.tool.darknet_shakedrop import DarknetShakedrop
            tmp = detector_config_file.split('/')
            detector_cfg = self.cfg.PERTURB.SHAKE_DROP.MODEL_CONFIG
            print('Self ensemble! Shake drop model cfg :', detector_config_file)
            tmp[-1] = detector_cfg
            self.detector = DarknetShakedrop('/'.join(tmp)).to(self.device)

            self.clean_model = Darknet(detector_config_file).to(self.device)
            self.clean_model.load_weights(model_weights)
            self.clean_model.eval()
            self.clean_model.models.requires_grad_(False)
        else:
            self.detector = Darknet(detector_config_file).to(self.device)

        self.detector.load_weights(model_weights)
        self.eval()


    def detect_img_batch_get_bbox_conf(self, batch_tensor, confs_thresh=0.5, clean_model=False):
        if clean_model and hasattr(self, 'clean_model'):
            output = self.clean_model(batch_tensor)
        else:
            output = self.detector(batch_tensor)
        preds = post_processing(batch_tensor, self.conf_thres, self.iou_thres, output)
        for i, pred in enumerate(preds):
            pred = np.array(pred)
            if len(pred) != 0:
                pred[:, :4] = np.clip(pred[:, :4], a_min=0, a_max=1)
            preds[i] = pred # shape([1, 6])
        # print('v4 h: ', confs.shape, confs.requires_grad)

        # output: [ [batch, num, 1, 4], [batch, num, num_classes] ]
        confs = output[1]
        confs = confs[torch.where(confs > confs_thresh)]

        return preds, confs
