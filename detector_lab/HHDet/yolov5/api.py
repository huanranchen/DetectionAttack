import torch

# load from YOLOV5
from .yolov5.utils.general import non_max_suppression, scale_coords
from .yolov5.models.experimental import attempt_load  # scoped to avoid circular import


from ...DetectorBase import DetectorBase


class HHYolov5(DetectorBase):
    def __init__(self, name, cfg,
                 input_tensor_size=640,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)
        self.imgsz = (input_tensor_size, input_tensor_size)
        self.stride, self.pt = None, None

    def load(self, model_weights, **args):
        w = str(model_weights[0] if isinstance(model_weights, list) else model_weights)
        self.detector = attempt_load(model_weights if isinstance(model_weights, list) else w,
                                     map_location=self.device, inplace=False)
        self.eval()
        self.stride = max(int(self.detector.stride.max()), 32)  # model stride
        self.names = self.detector.module.names if hasattr(self.detector, 'module') else self.detector.names  # get class names

    def detect_img_batch_get_bbox_conf(self, batch_tensor, confs_thresh=0.5, **kwargs):
        # print("detect: ", batch_tensor.requires_grad)
        output = self.detector(batch_tensor, augment=False, visualize=False)[0]
        # print("output       :", output[0].shape)
        preds = non_max_suppression(output.detach().cpu(),
                                    self.conf_thres, self.iou_thres) # [batch, num, 6] e.g., [1, 22743, 1, 4]
        bbox_array = []
        for pred in preds:
            # print(pred)
            box = scale_coords(batch_tensor.shape[-2:], pred, self.ori_size)
            box[:, [0,2]] /= self.ori_size[1]
            box[:, [1,3]] /= self.ori_size[0]
            bbox_array.append(box)

        # [batch, num, num_classes] e.g.,[1, 22743, 80]
        confs = output[..., 4]
        confs = confs[confs > confs_thresh]
        # print(bbox_array, confs)
        # print('filtered: ', confs.shape)
        return bbox_array, confs

