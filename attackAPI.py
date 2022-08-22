import sys

import cv2
import torch
import os
import numpy as np
import torch.nn.functional as F

from simgleAttack import DetctorAttacker
from tools.det_utils import rescale_patches
from tools import FormatConverter, DataTransformer, save_tensor
from tools.adv import PatchManager

import warnings
warnings.filterwarnings("ignore")


class UniversalDetectorAttacker(DetctorAttacker):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)

        self.patch_obj = PatchManager(cfg.ATTACKER.PATCH_ATTACK, device)
        if cfg.DATA.AUGMENT:
            self.data_transformer = DataTransformer(cfg.DETECTOR.INPUT_SIZE)

    def init_universal_patch(self, patch_file=None):
        self.patch_obj.init(patch_file)
        self.universal_patch = self.patch_obj.patch

    def filter_bbox(self, preds):
        filt = [pred[-1] in self.cfg.attack_list for pred in preds]
        preds = preds[filt]
        return np.array(preds)

    def get_patch_pos_batch(self, all_preds, aspect_ratio=None):
        # get all bboxs of setted target. If none target bbox is got, return has_target=False
        height, width = self.cfg.DETECTOR.INPUT_SIZE
        scale = self.cfg.ATTACKER.PATCH_ATTACK.SCALE
        self.batch_patch_boxes = []

        target_nums = []
        if aspect_ratio is None:
            aspect_ratio = self.cfg.ATTACKER.PATCH_ATTACK.ASPECT_RATIO

        for index, preds in enumerate(all_preds):
            # for every image in the batch
            # print('before filter: ', preds)
            if len(preds):
                preds = self.filter_bbox(preds)
                # print('filtered :', preds)
                patch_boxs = rescale_patches(preds, height, width, scale, aspect_ratio)
            else:
                patch_boxs = np.array([])
            # print(patch_boxs)

            self.batch_patch_boxes.append(patch_boxs)
            target_nums.append(len(patch_boxs))
        return np.array(target_nums)

    def apply_universal_patch(self, img_tensor, attacking=False, universal_patch=None):
        if universal_patch is None:
            universal_patch = self.universal_patch
        img_tensor = img_tensor.clone()

        universal_patch = universal_patch.detach_() if attacking else universal_patch.clone()
        # print('is leaf patch: ', universal_patch.is_leaf)
        if universal_patch.is_leaf:
            universal_patch.requires_grad = True

        for i, img in enumerate(img_tensor):
            for j, bbox in enumerate(self.batch_patch_boxes[i]):
                # for jth bbox in ith-img's bboxes
                p_x1, p_y1, p_x2, p_y2 = bbox[:4]
                height = p_y2 - p_y1
                width = p_x2 - p_x1
                if height <= 0 or width <= 0:
                    continue

                patch_tensor = F.interpolate(universal_patch, size=(height, width), mode='bilinear')
                img_tensor[i][:, p_y1:p_y2, p_x1:p_x2] = patch_tensor[0]

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

    def adv_detect_save(self, img_tensor, save_path, save_name, detectors=None):
        if detectors is None:
            detectors = self.detectors
        os.makedirs(save_path, exist_ok=True)
        for detector in detectors:
            print('adv_detect_save, apply universal patch')
            tmp, _ = self.apply_universal_patch(img_tensor[0].unsqueeze(0))
            preds, _ = detector.detect_img_batch_get_bbox_conf(tmp)
            img_numpy_int8 = FormatConverter.tensor2numpy_cv2(tmp[0].cpu().detach())
            self.plot_boxes(img_numpy_int8, preds[0],
                            savename=os.path.join(save_path, save_name))

    def save_patch(self, save_path, save_patch_name):
        save_path = save_path + '/patch/'
        # save_patch_name = os.path.join(save_path, save_patch_name)
        save_tensor(self.universal_patch, save_patch_name, save_path)
        print(self.universal_patch.is_leaf)
        print('Saving patch to ', os.path.join(save_path, save_patch_name))

    def attack_test(self, img_tensor_batch, optimizer):
        img_tensor_batch.requires_grad = True
        for detector in self.detectors:
            loss = self.attacker.sequential_non_targeted_attack(img_tensor_batch, self, detector, optimizer)
        return loss

    def attack(self, img_tensor_batch, mode='sequential', confs_thresh=None):
        # if self.cfg.DATA.AUGMENT:
        #     print(img_tensor_batch.shape)
        #     img_tensor_batch = self.data_transformer(img_tensor_batch)
        if mode == 'sequential':
            loss = self.sequential_attack(img_tensor_batch, confs_thresh)
            print("sequential attack loss ")
        elif mode == 'parallel':
            self.parallel_attack(img_tensor_batch)

    def sequential_attack(self, img_tensor_batch, confs_thresh=None):
        detectors_loss = []
        for detector in self.detectors:
            patch, loss = self.attacker.sequential_non_targeted_attack(
                img_tensor_batch, self, detector, confs_thresh=confs_thresh)
            self.patch_obj.update(patch)
            detectors_loss.append(loss)
        return np.mean(detectors_loss)

    def parallel_attack(self, img_tensor_batch):
        patch_updates = torch.zeros(self.universal_patch.shape).to(self.device)
        for detector in self.detectors:
            # adv_img_tensor = self.apply_universal_patch(img_numpy_batch)
            # print('------------------', detector.name)
            patch_update = self.attacker.parallel_non_targeted_attack(img_tensor_batch, self, detector)
            patch_updates += patch_update
        # print('self.universal: ', self.universal_patch)
        self.patch_obj.update(self.universal_patch + patch_updates / len(self.detectors))


