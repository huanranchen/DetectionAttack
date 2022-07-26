import torch
import numpy as np

from torchvision.models.detection import ssd300_vgg16
from ...DetectorBase import DetectorBase
# from .utils import inter_nms


class SSD(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_size=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)

    def load(self, model_weights=None, **args):
        kwargs = {}
        # if self.input_tensor_size is not None:
        #     kwargs['min_size'] = self.input_tensor_size
        self.detector = ssd300_vgg16(pretrained=True, **kwargs)

        self.detector = self.detector.to(self.device)
        self.detector.eval()
        self.detector.requires_grad_(False)

    def detect_img_batch_get_bbox_conf(self, batch_tensor, confs_thresh=0.5):
        shape = batch_tensor.shape[-2]
        preds = self.detector(batch_tensor)
        # print(confs[0])
        confs_array = None
        bbox_array = []
        for ind, pred in enumerate(preds):
            array = None
            for box, cls, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                # cls-1: ignore the background class
                tmp = torch.cat((box/shape, score.unsqueeze(0), (cls-1).unsqueeze(0)), -1)
                array = tmp if array is None else torch.vstack((array, tmp))
            conf = pred['scores']
            print(cls.shape, conf.shape)
            ctmp = conf[conf > confs_thresh]
            confs_array = ctmp if confs_array is None else torch.cat((confs_array, ctmp), -1)

            if array is None:
                array = np.array([])
            else:
                array = array.detach().cpu()
            bbox_array.append(array)

        # bbox_array = inter_nms(bbox_array, self.conf_thres, self.iou_thres)

        return bbox_array, confs_array