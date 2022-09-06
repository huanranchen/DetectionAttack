import sys

import torch
from . import ssd300_vgg16
from ...base import DetectorBase
from .. import inter_nms


class TorchSSD(DetectorBase):
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

    def __call__(self, batch_tensor, **kwargs):
        shape = batch_tensor.shape[-2]
        preds = self.detector(batch_tensor)

        # print(confs[0])
        cls_max_ids = None
        confs_array = None
        bbox_array = []
        for ind, pred in enumerate(preds):
            len = pred['scores'].shape[0]
            array = torch.cat((
                pred['boxes']/shape,
                pred['scores'].view(len, 1),
                (pred['labels']-1).view(len, 1)
            ), 1) if len else torch.cuda.FloatTensor([])

            conf = pred['scores']
            confs_array = conf if confs_array is None else torch.cat((confs_array, conf), -1)
            bbox_array.append(array)

        bbox_array = inter_nms(bbox_array, self.conf_thres, self.iou_thres)
        output = {'bbox_array': bbox_array, 'obj_confs': confs_array, "cls_max_ids": cls_max_ids}
        return output