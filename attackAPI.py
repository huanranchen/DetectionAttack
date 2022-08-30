import sys

import cv2
import torch
import os
import numpy as np
import torch.nn.functional as F

from simgleAttack import DetctorAttacker
from tools.det_utils import rescale_patches
from tools import FormatConverter, DataTransformer, save_tensor
from tools.adv import PatchManager, PatchRandomApplier
from detector_lab.utils import inter_nms

import warnings
warnings.filterwarnings("ignore")


class UniversalDetectorAttacker(DetctorAttacker):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.vlogger = None
        self.patch_obj = PatchManager(cfg.ATTACKER.PATCH_ATTACK, device)
        self.patch_apply = PatchRandomApplier(device, scale_rate=cfg.ATTACKER.PATCH_ATTACK.SCALE)
        if cfg.DATA.AUGMENT > 1:
            self.data_transformer = DataTransformer(device)

    @property
    def universal_patch(self):
        return self.patch_obj.patch

    def init_universal_patch(self, patch_file=None):
        self.patch_obj.init(patch_file)
        # self.universal_patch = self.patch_obj.patch

    def filter_bbox(self, preds, cls_array=None):
        if cls_array is None:
            cls_array = preds[:, -1]
        filt = [cls in self.cfg.attack_list for cls in cls_array]
        preds = preds[filt]
        return preds

    def get_patch_pos_batch(self, all_preds, aspect_ratio=None):
        # get all bboxs of setted target. If none target bbox is got, return has_target=False
        height, width = self.cfg.DETECTOR.INPUT_SIZE
        scale = self.cfg.ATTACKER.PATCH_ATTACK.SCALE
        self.all_preds = all_preds
        self.batch_patch_boxes = []

        target_nums = []
        if aspect_ratio is None:
            aspect_ratio = self.cfg.ATTACKER.PATCH_ATTACK.ASPECT_RATIO

        for i_batch, preds in enumerate(all_preds):
            if len(preds):
                # print(preds)
                preds = self.filter_bbox(preds)
                patch_boxs = rescale_patches(preds, height, width, scale, aspect_ratio)
                # print(patch_boxs)
            else:
                patch_boxs = torch.Tensor([]).to(preds.device)
            self.batch_patch_boxes.append(patch_boxs)
            target_nums.append(len(patch_boxs))
        return np.array(target_nums)

    def uap_apply(self, img_tensor, universal_patch=None, eval=False):
        if universal_patch is None:
            universal_patch = self.universal_patch

        if eval:
            for i in range(img_tensor.shape[0]):
                for j, bbox in enumerate(self.batch_patch_boxes[i]):
                    # for jth bbox in ith-img's bboxes
                    p_x1, p_y1, p_x2, p_y2 = bbox[:4]
                    height = p_y2 - p_y1
                    width = p_x2 - p_x1
                    if height <= 0 or width <= 0:
                        continue
                    if self.cfg.DATA.AUGMENT == 3:
                        adv = self.data_transformer.jitter(universal_patch)
                    else:
                        adv = universal_patch
                    adv = F.interpolate(adv, size=(height, width), mode='bilinear')[0]
                    # Warning: This is an inplace operation, it changes the value of the outer variable
                    img_tensor[i][:, p_y1:p_y2, p_x1:p_x2] = adv
        else:
            import time
            t_time0 = time.time()
            img_tensor = self.patch_apply(img_tensor, universal_patch, self.all_preds)
            t_time1 = time.time()
            print('t_time: ', t_time1-t_time0)

        if self.cfg.DATA.AUGMENT == 2:
            img_tensor = self.data_transformer(img_tensor, True, True)
        # if self.cfg.DATA.AUGMENT == 3:
        #     img_tensor = self.data_transformer(img_tensor, False, True)

        return img_tensor

    def merge_batch_pred(self, all_preds, preds):
        if all_preds is None:
            return preds
        for i, (all_pred, pred) in enumerate(zip(all_preds, preds)):
            if pred.shape[0]:
                all_preds[i] = torch.cat((all_pred, pred), dim=0)
                continue
            all_preds[i] = all_pred if all_pred.shape[0] else pred

        return all_preds

    def adv_detect_save(self, img_tensor, save_path, save_name, detector):
        tmp = self.uap_apply(img_tensor, eval=True)
        preds = detector(tmp)['bbox_array']
        self.plot_boxes(tmp[0], preds[0], save_path, savename=save_name)

    def detect_bbox(self, img_batch, detectors=None):
        if detectors is None:
            detectors = self.detectors

        all_preds = None
        for detector in detectors:
            preds = detector(img_batch)['bbox_array']
            all_preds = self.merge_batch_pred(all_preds, preds)

        # nms among detectors
        # print(all_preds)
        if len(detectors) > 1: all_preds = inter_nms(all_preds)
        # print(all_preds)
        return all_preds

    def attack(self, img_tensor_batch, mode='sequential'):
        if mode == 'optim':
            for detector in self.detectors:
                loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
        else:
            if mode == 'sequential':
                loss = self.sequential_attack(img_tensor_batch)
                print("sequential attack loss ", loss)
            elif mode == 'parallel':
                loss = self.parallel_attack(img_tensor_batch)
        return loss

    def sequential_attack(self, img_tensor_batch):
        detectors_loss = []
        for detector in self.detectors:
            patch, loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
            self.patch_obj.update_(patch.detach_())
            # print('sequential attack: ', self.universal_patch.is_leaf)
            detectors_loss.append(loss)
        return torch.tensor(detectors_loss).mean()

    def parallel_attack(self, img_tensor_batch):
        detectors_loss = []
        patch_updates = torch.zeros(self.universal_patch.shape).to(self.device)
        for detector in self.detectors:
            patch_tmp, loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
            patch_update = patch_tmp - self.universal_patch
            patch_updates += patch_update
            detectors_loss.append(loss)
        self.patch_obj.update_((self.universal_patch + patch_updates / len(self.detectors)).detach_())
        return torch.tensor(detectors_loss).mean()


