import numpy as np

from detector_lab.utils import init_detectors
from attacks.bim import LinfBIMAttack
from attacks.mim import LinfMIMAttack
from attacks.pgd import LinfPGDAttack
from attacks.optim import OptimAttacker
from tools.loss import *
from tools.det_utils import plot_boxes_cv2, scale_area_ratio

attacker_dict = {
    "bim": LinfBIMAttack,
    "mim": LinfMIMAttack,
    "pgd": LinfPGDAttack,
    "optim": OptimAttacker
}

loss_dict = {
    "default": temp_attack_loss,
    "ascend-mse": ascend_mse_loss,
    "descend-mse": descend_mse_loss,
    "obj-tv": obj_tv_loss
}


class DetctorAttacker(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.detectors = init_detectors(self.cfg.DETECTOR.NAME, cfg)
        self.patch_boxes = []
        self.class_names = cfg.all_class_names # class names reference: labels of all the classes
        self.attack_list = cfg.attack_list # int list: classes index to be attacked, [40, 41, 42, ...]

        self.init_attaker()

    def init_attaker(self):
        cfg = self.cfg.ATTACKER
        loss_func = loss_dict['default']
        if hasattr(cfg, 'LOSS_FUNC') and cfg.LOSS_FUNC is not None:
            loss_func = loss_dict[cfg.LOSS_FUNC]

        self.attacker = attacker_dict[cfg.METHOD](
            loss_func=loss_func, norm='L_infty', device=self.device, cfg = cfg, detector_attacker=self)


    def plot_boxes(self, img, boxes, savename=None):
        # print(img.dtype, isinstance(img, np.ndarray))
        plot_boxes_cv2(img, boxes.cpu().detach().numpy(), self.class_names, savename=savename)

    def patch_pos(self, preds, aspect_ratio=-1):
        height, width = self.cfg.DETECTOR.INPUT_SIZE
        scale_rate = self.cfg.ATTACKER.PATCH_ATTACK.SCALE
        patch_boxs = []

        # print(preds.shape)
        for pred in preds:
            # print('get pos', pred)
            x1, y1, x2, y2, conf, id = pred
            # print('bbox: ', x1, y1, x2, y2)

            # cfg.ATTACKER.ATTACK_CLASS have been processed to an int list
            if -1 in self.attack_list or int(id) in self.attack_list:
                p_x1, p_y1, p_x2, p_y2 = scale_area_ratio(
                    x1, y1, x2, y2, height, width, scale_rate, aspect_ratio)
                patch_boxs.append([p_x1, p_y1, p_x2, p_y2])
                # print('rectify bbox:', [p_x1, p_y1, p_x2, p_y2])
        return patch_boxs


    def get_patch_pos(self, preds, patch):
        patch_boxs = self.patch_pos(preds, patch)
        self.patch_boxes += patch_boxs
        # print("self.patch_boxes", patch_boxs)

    def init_patches(self):
        for i in range(len(self.patch_boxes)):
            p_x1, p_y1, p_x2, p_y2 = self.patch_boxes[i][:4]
            patch = np.random.randint(low=0, high=255, size=(p_y2 - p_y1, p_x2 - p_x1, 3))
            self.patch_boxes[i].append(patch)  # x1, y1, x2, y2, patch


    def apply_patches(self, img_tensor, detector, is_normalize=True):
        for i in range(len(self.patch_boxes)):
            p_x1, p_y1, p_x2, p_y2 = self.patch_boxes[i][:4]
            if is_normalize:
                # init the patches
                patch_tensor = detector.normalize(self.patch_boxes[i][-1])
            else:
                # when attacking
                patch_tensor = self.patch_boxes[i][-1].detach()
                patch_tensor.requires_grad = True
            self.patch_boxes[i][-1] = patch_tensor  # x1, y1, x2, y2, patch_tensor
            img_tensor[0, :, p_y1:p_y2, p_x1:p_x2] = patch_tensor
        return img_tensor