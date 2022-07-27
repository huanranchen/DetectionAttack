import sys

import torch
import math
import os
import numpy as np
import torch.nn.functional as F

from detector_lab.utils import init_detectors
from attacks.bim import LinfBIMAttack
from attacks.mim import LinfMIMAttack
from attacks.pgd import LinfPGDAttack
from losses import temp_attack_loss
from tools.utils import scale_area_ratio, transform_patch
from tools.det_utils import plot_boxes_cv2


attacker_dict = {
    "bim": LinfBIMAttack,
    "mim": LinfMIMAttack,
    "pgd": LinfPGDAttack,
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
        cfg = self.cfg
        self.attacker = attacker_dict[cfg.ATTACKER.METHOD](loss_fuction=temp_attack_loss, model=self.detectors,
                                                           norm='L_infty',
                                                           epsilons=cfg.ATTACKER.EPSILON,
                                                           max_iters=cfg.ATTACKER.MAX_ITERS,
                                                           step_size=cfg.ATTACKER.STEP_SIZE,
                                                           class_id=cfg.ATTACKER.TARGET_CLASS)
        for detector in self.detectors:
            self.attacker.init_epsilon(detector)


    def plot_boxes(self, img, boxes, savename=None):
        # print(img.dtype, isinstance(img, np.ndarray))
        plot_boxes_cv2(img, np.array(boxes), self.class_names, savename=savename)

    def patch_pos(self, preds, patch):
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
                p_x1, p_y1, p_x2, p_y2 = scale_area_ratio(x1, y1, x2, y2, height, width, scale_rate)
                patch_boxs.append([p_x1, p_y1, p_x2, p_y2])
                # print('rectify bbox:', [p_x1, p_y1, p_x2, p_y2])
        # transform_patch(preds, patch, scale_rate)
        # sys.exit()
        return patch_boxs

    def patch_aspect_ratio(self):
        height, width = self.universal_patch.shape[-2:]
        return width/height

    def get_patch_pos(self, preds, img_cv2, patch):
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


class UniversalDetectorAttacker(DetctorAttacker):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.universal_patch = None

    def init_universal_patch(self):
        height = self.cfg.ATTACKER.PATCH_ATTACK.HEIGHT
        width = self.cfg.ATTACKER.PATCH_ATTACK.WIDTH
        universal_patch = np.random.randint(low=0, high=255, size=(height, width, 3))
        universal_patch = np.expand_dims(np.transpose(universal_patch, (2, 0, 1)), 0)
        self.universal_patch = torch.from_numpy(np.array(universal_patch, dtype='float32') / 255.).to(self.device)
        # self.universal_patch.requires_grad = True

    def get_patch_pos_batch(self, all_preds):
        # get all bboxs of setted target. If none target bbox is got, return has_target=False
        # img_cv2: [b, h, w, c]
        self.batch_patch_boxes = []
        target_nums = []
        # has_target = False
        # print(self.universal_patch.shape)

        for index, preds in enumerate(all_preds):
            # for every image in the batch
            patch_boxs = self.patch_pos(preds, self.universal_patch)
            self.batch_patch_boxes.append(patch_boxs)
            target_nums.append(len(patch_boxs))
        return target_nums

    def apply_universal_patch(self, img_tensor, detector, is_normalize=True, universal_patch=None):
        if universal_patch is None:
            universal_patch = self.universal_patch
        # print("fn0: ", img_tensor.grad_fn)
        img_tensor = detector.normalize_tensor(img_tensor)

        if is_normalize:
            # init the patches
            universal_patch = detector.normalize_tensor(universal_patch)
        else:
            # when attacking
            universal_patch = universal_patch.detach_()
        if universal_patch.is_leaf:
            universal_patch.requires_grad = True
        # print(len(img_tensor))
        # print(self.detectors[0].name, "universal patch grad: ", universal_patch.requires_grad, universal_patch.is_leaf)
        for i, img in enumerate(img_tensor):
            # print('adding patch, ', i, 'got box num:', len(self.batch_patch_boxes[i]))
            for j, bbox in enumerate(self.batch_patch_boxes[i]):
                # for jth bbox in ith-img's bboxes
                p_x1, p_y1, p_x2, p_y2 = bbox[:4]
                height = p_y2 - p_y1
                width = p_x2 - p_x1
                if height <= 0 or width <= 0:
                    continue
                patch_tensor = F.interpolate(universal_patch, size=(height, width), mode='bilinear')
                # print("fn1: ", img_tensor.grad_fn, patch_tensor.grad_fn)
                img_tensor[i][:, p_y1:p_y2, p_x1:p_x2] = patch_tensor[0]
                # print("fn2: ", img_tensor.grad_fn)

        return img_tensor, universal_patch

    def merge_batch_pred(self, all_preds, preds):
        if all_preds is None:
            # if no box detected, that dimention will be a array([])
            all_preds = preds
            # print(preds[0].shape)
        else:
            for i, (all_pred, pred) in enumerate(zip(all_preds, preds)):
                pred = np.array(pred)
                if all_pred.shape[0] and pred.shape[0]:
                    all_preds[i] = np.r_[all_pred, pred]
                    continue
                all_preds[i] = all_pred if all_pred.shape[0] else pred

        return all_preds

    def imshow_save(self, img_tensor, save_path, save_name, detectors=None):
        if detectors is None:
            detectors = self.detectors
        os.makedirs(save_path, exist_ok=True)
        for detector in detectors:
            tmp, _ = self.apply_universal_patch(img_tensor, detector)
            preds, _ = detector.detect_img_batch_get_bbox_conf(tmp)
            img_numpy, img_numpy_int8 = detector.unnormalize(tmp[0])
            self.plot_boxes(img_numpy_int8, preds[0],
                            savename=os.path.join(save_path, save_name))

    def save_patch(self, save_path, save_patch_name, patch=None, pth=False):
        import cv2
        save_path = save_path + '/patch/'
        os.makedirs(save_path, exist_ok=True)
        save_patch_name = os.path.join(save_path, save_patch_name)
        if patch is None:
            patch = self.universal_patch.clone()

        if pth:
            torch.save(patch, save_patch_name)
        else:
            tmp = patch.cpu().detach().squeeze(0).numpy()
            tmp *= 255.
            tmp = np.transpose(tmp, (1, 2, 0))
            tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
            tmp = tmp.astype(np.uint8)
            # print(tmp.shape)
            # cv2.imwrite('./patch.png', tmp)
            cv2.imwrite(save_patch_name, tmp)
        print('saving patch to ' + save_patch_name)

    def serial_attack(self, img_tensor_batch, confs_thresh=None):
        for detector in self.detectors:
            patch = self.attacker.serial_non_targeted_attack(
                img_tensor_batch, self, detector, confs_thresh=confs_thresh)
            self.universal_patch = patch

    def parallel_attack(self, img_tensor_batch):
        patch_updates = torch.zeros(self.universal_patch.shape).to(self.device)
        for detector in self.detectors:
            # adv_img_tensor = self.apply_universal_patch(img_numpy_batch, detector)
            # print('------------------', detector.name)
            patch_update = self.attacker.parallel_non_targeted_attack(img_tensor_batch, self, detector)
            patch_updates += patch_update
        # print('self.universal: ', self.universal_patch)
        self.universal_patch += patch_updates / len(self.detectors)

