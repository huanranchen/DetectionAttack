import sys

import cv2
import torch
import os
import numpy as np
import torch.nn.functional as F

from simgleAttack import DetctorAttacker
from tools.det_utils import rescale_patches

import warnings
warnings.filterwarnings("ignore")


class UniversalDetectorAttacker(DetctorAttacker):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.universal_patch = None

    def read_patch(self, patch_file):
        # patch_file = self.args.patch
        print('Reading patch from file: ' + patch_file)
        if patch_file.endswith('.pth'):
            universal_patch = torch.load(patch_file, map_location=self.device).unsqueeze(0)
            # universal_patch.new_tensor(universal_patch)
            print(universal_patch.shape, universal_patch.requires_grad, universal_patch.is_leaf)
        else:
            universal_patch = cv2.imread(patch_file)
            universal_patch = cv2.cvtColor(universal_patch, cv2.COLOR_BGR2RGB)
            universal_patch = np.expand_dims(np.transpose(universal_patch, (2, 0, 1)), 0)
            universal_patch = torch.from_numpy(np.array(universal_patch, dtype='float32') / 255.)
        self.universal_patch = universal_patch.to(self.device)

    def init_universal_patch(self, patch_file=None, init_mode='random'):
        if patch_file is None:
            self.generate_universal_patch(init_mode)
        else:
            self.read_patch(patch_file)

    def generate_universal_patch(self, init_mode='random'):
        height = self.cfg.ATTACKER.PATCH_ATTACK.HEIGHT
        width = self.cfg.ATTACKER.PATCH_ATTACK.WIDTH
        if init_mode.lower() == 'random':
            print('Random initializing a universal patch')
            universal_patch = np.random.randint(low=0, high=255, size=(3, height, width))
        elif init_mode.lower() == 'gray':
            print('Gray initializing a universal patch')
            universal_patch = np.ones((3, height, width)) * 127.5

        universal_patch = np.expand_dims(universal_patch, 0)
        self.universal_patch = torch.from_numpy(np.array(universal_patch, dtype='float32') / 255.).to(self.device)
        # self.universal_patch.requires_grad = True

    def filter_bbox(self, preds):
        # print('attack list: ', self.cfg.attack_list)
        # print('preds: ', preds)
        filt = []
        for pred in preds:
            # print(pred[-1], pred[-1] in self.cfg.attack_list)
            filt.append(pred[-1] in self.cfg.attack_list)
        # preds = list(filter(lambda bbox: (bbox[-1] in self.cfg.attack_list), preds))
        preds = preds[filt]
        # print(preds)
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
        # print('====================is leaf: ', universal_patch.is_leaf)
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
            tmp, _ = self.apply_universal_patch(img_tensor[0].unsqueeze(0), detector)
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

    def attack_test(self, img_tensor_batch, optimizer):
        img_tensor_batch.requires_grad = True
        for detector in self.detectors:
            loss = self.attacker.serial_non_targeted_attack(img_tensor_batch, self, detector, optimizer)
        return loss

    def serial_attack(self, img_tensor_batch, confs_thresh=None):
        detectors_loss = []
        for detector in self.detectors:
            patch, loss = self.attacker.serial_non_targeted_attack(
                img_tensor_batch, self, detector, confs_thresh=confs_thresh)
            self.universal_patch = patch
            detectors_loss.append(loss)
        return np.mean(detectors_loss)

    def parallel_attack(self, img_tensor_batch):
        patch_updates = torch.zeros(self.universal_patch.shape).to(self.device)
        for detector in self.detectors:
            # adv_img_tensor = self.apply_universal_patch(img_numpy_batch, detector)
            # print('------------------', detector.name)
            patch_update = self.attacker.parallel_non_targeted_attack(img_tensor_batch, self, detector)
            patch_updates += patch_update
        # print('self.universal: ', self.universal_patch)
        self.universal_patch += patch_updates / len(self.detectors)

