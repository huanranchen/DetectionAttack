import numpy as np

from .Pytorch_YOLOv4.tool.utils import *
from .Pytorch_YOLOv4.tool.torch_utils import *
from .Pytorch_YOLOv4.tool.darknet2pytorch import Darknet

from ...DetectorBase import DetectorBase


class HHYolov4(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_size=416,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        # name
        super().__init__(name, cfg, input_tensor_size, device)

    def load(self, model_weights, cfg_file=None, data_config_path=None):
        self.detector = Darknet(cfg_file)

        # data config
        # self.num_classes = self.detector.num_classes
        # self.namesfile = load_class_names(self.cfg.CLASS_NAME_FILE)

        # post_processing method | input (img, conf_thresh, nms_thresh, output) | return bboxes_batch
        self.post_processing = post_processing

        self.detector.load_weights(model_weights)
        self.detector.eval()
        self.detector.to(self.device)

        self.module_list = self.detector.models

        self.module_list.requires_grad_(False)


    # def init_img_batch(self, img_numpy_batch):
    #     img_tensor = Variable(torch.from_numpy(img_numpy_batch).float().div(255.0).to(self.device))
    #     return img_tensor

    def detect_img_batch_get_bbox_conf(self, batch_tensor, confs_thresh=0.5):
        output = self.detector(batch_tensor)
        # output: [ [batch, num, 1, 4], [batch, num, num_classes] ]
        confs = output[1]
        # print(self.name, confs.shape)
        preds = self.post_processing(batch_tensor, self.conf_thres, self.iou_thres, output)
        for i, pred in enumerate(preds):
            pred = np.array(pred)
            if len(pred) != 0:
                pred[:, :4] = np.clip(pred[:, :4], a_min=0, a_max=1)
            preds[i] = pred # shape([1, 6])
        # print('v4 h: ', confs.shape, confs.requires_grad)

        confs = confs[torch.where(confs > confs_thresh)]

        return preds, confs
