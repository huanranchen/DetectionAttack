import sys
import cv2
import torch
import os
import numpy as np
import torch.nn.functional as F

from simgleAttack import DetctorAttacker
from tools import FormatConverter, DataTransformer, save_tensor, attach_patch
from tools.adv import PatchManager, PatchRandomApplier
from detector_lab.utils import inter_nms

import warnings

warnings.filterwarnings("ignore")


class UniversalDetectorAttacker(DetctorAttacker):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.vlogger = None
        self.gates = {'jitter': False, 'median_pool': False, 'rotate': True, 'shift': False, 'p9_scale': False}
        self.patch_obj = PatchManager(cfg.ATTACKER.PATCH_ATTACK, device)
        self.patch_apply = PatchRandomApplier(device, scale_rate=cfg.ATTACKER.PATCH_ATTACK.SCALE)
        self.max_boxes = 15
        if '3' in cfg.DATA.AUGMENT:
            self.data_transformer = DataTransformer(device)

    @property
    def universal_patch(self):
        return self.patch_obj.patch

    def init_universal_patch(self, patch_file=None):
        self.patch_obj.init(patch_file)
        # self.universal_patch = self.patch_obj.patch

    def filter_bbox(self, preds, cls_array=None):
        # print(preds)
        if cls_array is None: cls_array = preds[:, -1]
        filt = [cls in self.cfg.attack_list for cls in cls_array]
        preds = preds[filt]
        return preds

    def get_patch_pos_batch(self, all_preds, aspect_ratio=None):
        # get all bboxs of setted target. If none target bbox is got, return has_target=False
        self.all_preds = all_preds
        batch_boxes = None
        target_nums = []
        for i_batch, preds in enumerate(all_preds):
            if len(preds) == 0:
                preds = torch.cuda.FloatTensor([[0, 0, 0, 0, 0, 0]])
            preds = self.filter_bbox(preds)[:self.max_boxes]
            pad_size = self.max_boxes - len(preds)
            padded_boxs = F.pad(preds, (0, 0, 0, pad_size), value=0).unsqueeze(0)
            batch_boxes = padded_boxs if batch_boxes is None else torch.vstack((batch_boxes, padded_boxs))
            target_nums.append(len(preds))
        self.all_preds = batch_boxes
        return np.array(target_nums)

    def uap_apply(self, img_tensor, adv_patch=None, gates=None):
        if adv_patch is None: adv_patch = self.universal_patch
        if gates is None: gates = self.gates

        img_tensor = self.patch_apply(img_tensor, adv_patch, self.all_preds, gates=gates)
        if '2' in self.cfg.DATA.AUGMENT: img_tensor = self.data_transformer(img_tensor, True, True)
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
            elif mode == 'parallel':
                loss = self.parallel_attack(img_tensor_batch)
        return loss

    def sequential_attack(self, img_tensor_batch):
        detectors_loss = []
        for detector in self.detectors:
            loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
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
