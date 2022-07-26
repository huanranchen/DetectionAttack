import torch

from .pytorch_faster_rcnn.utils.train_utils import create_model

from ...DetectorBase import DetectorBase
from tools.parser import load_class_names


class Faster_RCNN(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_size=640,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)

    def load(self, model_weights, **args):
        self.namesfile = load_class_names(self.cfg.CLASS_NAME_FILE)
        self.detector = create_model(num_classes=len(self.namesfile)+1).to(self.device)
        checkpoint = torch.load(model_weights, map_location='cpu')
        self.detector.load_state_dict(checkpoint['model'])
        self.detector.eval()
        # print(self.name)
        # print(self.detector)
        pass

    def detect_img_batch_get_bbox_conf(self, batch_tensor):
        print(batch_tensor.shape)
        preds = self.detector(batch_tensor)
        for pred in preds:
            print(pred['boxes'])
            # print(pred['labels'])
            # print(pred['scores'])
        pass

