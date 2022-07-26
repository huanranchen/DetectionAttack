import torch

from .ssd_pytorch.ssd import build_ssd
# from ... import load_class_names
from ...DetectorBase import DetectorBase


class HHSSD(DetectorBase):
    def __init__(self, name, cfg,
                 input_size=416,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        # name
        super().__init__(name, cfg, input_size, device)

    def load(self, model_weights, **args):
        self.load_class_names(self.cfg.CLASS_NAME_FILE)
        # classes + 1 dustbin class
        self.detector = build_ssd('test', 300, len(self.class_names) + 1)            # initialize SSD
        self.detector.load_state_dict(torch.load(model_weights))
        self.detector.eval()

    def detect_img_batch_get_bbox_conf(self, batch_tensor):
        detections = self.detector(batch_tensor)
        print(detections.data)
        pass