import copy
import sys

from .base import BaseAttacker

import torch
import torch.distributed as dist
import numpy as np

num_iter = 0
update_pre = 0
# patch_tmp = None

import sys
from pathlib import Path
PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.append(PROJECT_ROOT)
from tools import FormatConverter


class LinfPGDAttack(BaseAttacker):
    """PGD attacks (arxiv: https://arxiv.org/pdf/1706.06083.pdf)"""
    def __init__(self, loss_func, cfg, device, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)

    def patch_update(self, **kwargs):
        # print('before: ', self.detector_attacker.patch_obj.patch.sum())
        patch_tmp = self.detector_attacker.patch_obj.patch
        update = self.step_size * patch_tmp.grad.sign()
        if "descend" in self.cfg.LOSS_FUNC:
            update *= -1
        patch_tmp = patch_tmp + update
        patch_tmp = torch.clamp(patch_tmp, min=self.min_epsilon, max=self.max_epsilon)
        self.detector_attacker.patch_obj.update_(patch_tmp)
        # print(self.detector_attacker.patch_obj.patch.sum())
        # patch_clamp_(p_min=self.min_epsilon, p_max=self.max_epsilon)
        return patch_tmp

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []
        for iter in range(self.iter_step):
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch.clone(), mode=self.cfg.METHOD)
            bboxes, confs, cls_array = detector(adv_tensor_batch).values() # detect adv img batch to get bbox and obj confs
            confs = confs.max(dim=-1, keepdim=True)[0]
            # bbox_num = torch.FloatTensor([len(pred) for pred in preds])
            # if torch.sum(bbox_num) == 0: break
            detector.zero_grad()
            loss = self.attack_loss(confs)
            loss.backward()
            self.patch_update()
            losses.append(float(loss))
        self.logger(detector, adv_tensor_batch, bboxes)
        return torch.cuda.FloatTensor(losses).mean()

    def attack_loss(self, confs):
        return self.loss_fn(confs=confs)

    def parallel_non_targeted_attack(self, ori_tensor_batch, detector_attacker, detector):
        adv_tensor_batch, patch_tmp = detector_attacker.uap_apply(ori_tensor_batch)
        loss = []
        for iter in range(detector_attacker.cfg.ATTACKER.ITER_STEP):
            preds, confs = detector(adv_tensor_batch)
            disappear_loss = self.attack_loss(confs)
            loss.append(float(disappear_loss))
            detector.zero_grad()
            disappear_loss.backward()
            self.patch_update(patch_tmp, detector_attacker.patch_obj.clamp_)
            adv_tensor_batch, _ = detector_attacker.uap_apply(ori_tensor_batch, universal_patch=patch_tmp)

        return patch_tmp
            