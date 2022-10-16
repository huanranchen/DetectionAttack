import numpy as np
import torch
from .base import BaseAttacker
from torchvision.transforms import functional as TF
from torch.nn import functional as F
import random


def symmetrical_loss(x, scale_factor=0.1):
    '''
    improving the symmetry of the input tensor
    '''
    horizontal_flipped = TF.hflip(x.detach())
    vertical_flipped = TF.vflip(x.detach())
    loss = F.mse_loss(x, horizontal_flipped) + F.mse_loss(x, vertical_flipped)
    return scale_factor * loss


def horizontal_shift_invariant_loss(x, min_translate_ratio=0.2, scale_factor=0.1):
    '''
    for example:
    if x is [300, 300] vector, min_translate_ratio is 0.2, the translation may be:
    0.2*300, 0.4*300, 0.6*300, 0.8*300, 0/1 * 300
    '''
    assert len(x.shape) == 3, 'must be C, H, W shape'
    this_translate_ratio = 1 / random.randint(0, int(1 / min_translate_ratio))
    translate_threshold = x.shape[1] * this_translate_ratio
    translated = torch.zeros_like(x.detach())
    translated[:, -translate_threshold:, :] = x[:, :translate_threshold, :]
    translated[:, :-translate_threshold, :] = x[:, translate_threshold:, :]
    loss = F.mse_loss(x, translated)
    return scale_factor * loss


class PriorAttacker(BaseAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def patch_update(self, **kwargs):
        self.optimizer.step()
        self.patch_obj.clamp_(p_min=self.min_epsilon, p_max=self.max_epsilon)

    def attack_loss(self, confs):
        self.optimizer.zero_grad()
        loss = self.loss_fn(confs=confs, patch=self.detector_attacker.universal_patch[0])
        tv_loss, obj_loss = loss.values()
        tv_loss = torch.max(self.cfg.tv_eta * tv_loss, torch.tensor(0.0).to(self.device))
        # print(obj_loss)
        obj_loss = obj_loss * self.cfg.obj_eta
        loss = tv_loss + obj_loss
        out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss}
        return out
