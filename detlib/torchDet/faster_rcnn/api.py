import torch

from . import faster_rcnn, fasterrcnn_resnet50_fpn
from ...base import DetectorBase
from .. import inter_nms


class TorchFasterRCNN(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_size=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)

    def load(self, model_weights=None, **args):
        kwargs = {}
        if self.input_tensor_size is not None:
            kwargs['min_size'] = self.input_tensor_size
        if self.cfg.PERTURB.GATE == 'shake_drop':
            from .faster_rcnn import faster_rcnn_resnet50_shakedrop
            self.detector = faster_rcnn_resnet50_shakedrop()
        else:
            self.detector = fasterrcnn_resnet50_fpn(pretrained=True, **kwargs) \
                if model_weights is None else fasterrcnn_resnet50_fpn()

        self.detector = self.detector.to(self.device)
        self.eval()

    def __call__(self, batch_tensor, **kwargs):
        shape = batch_tensor.shape[-2]
        preds, confs = self.detector(batch_tensor)

        cls_max_ids = None
        confs_array = []
        bbox_array = []
        for ind, (pred, now_confidence) in enumerate(zip(preds, confs)):
            nums = pred['scores'].shape[0]
            array = torch.cat((
                pred['boxes'] / shape,
                pred['scores'].view(nums, 1),
                (pred['labels'] - 1).view(nums, 1)
            ), 1) if nums else torch.cuda.FloatTensor([])
            confs_array.append(now_confidence)
            bbox_array.append(array)

        confs_array = torch.cat(confs_array, dim = 0)

        bbox_array = inter_nms(bbox_array, self.conf_thres, self.iou_thres)
        output = {'bbox_array': bbox_array, 'obj_confs': confs_array, "cls_max_ids": cls_max_ids}
        return output