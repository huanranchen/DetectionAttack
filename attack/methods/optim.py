import numpy as np
import torch
from .base import BaseAttacker
from torch.optim import Optimizer


class OptimAttacker(BaseAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer
        if self.detector_attacker.vlogger is not None:
            self.detector_attacker.vlogger.optimizer = optimizer

    def patch_update(self, **kwargs):
        self.optimizer.step()
        # grad = self.optimizer.param_groups[0]['params'][0].grad
        # print(torch.mean(torch.abs(grad)))
        self.patch_obj.clamp_(p_min=self.min_epsilon, p_max=self.max_epsilon)

    def attack_loss(self, confs):
        self.optimizer.zero_grad()
        loss = self.loss_fn(confs=confs, patch=self.detector_attacker.universal_patch[0])
        tv_loss, obj_loss = loss.values()
        tv_loss = torch.max(self.cfg.tv_eta * tv_loss, torch.tensor(0.1).to(self.device))
        # print(obj_loss)
        obj_loss = obj_loss * self.cfg.obj_eta
        loss = tv_loss + obj_loss
        out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss}
        return out
