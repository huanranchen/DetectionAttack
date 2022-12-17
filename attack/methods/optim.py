import numpy as np
import torch
from .base import BaseAttacker
from torch.optim import Optimizer


class OptimAttacker(BaseAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)

    # @property
    # def param_groups(self):
    #     return self.optimizer.param_groups

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

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
        loss = tv_loss.to(obj_loss.device) + obj_loss
        out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss}
        return out


class SAMAttacker(BaseAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty', reverse_step_size=1e-2):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)
        self.reverse_step_size = reverse_step_size

    # @property
    # def param_groups(self):
    #     return self.optimizer.param_groups

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer
        self.iter = 0

    @torch.no_grad()
    def patch_update(self, **kwargs):
        self.iter += 1
        if self.iter % 2:
            self.patch_obj.patch += self.reverse_step_size * \
                                    self.patch_obj.patch.grad / torch.norm(self.patch_obj.patch.grad, p=2)
            self.iter = 0
        else:
            self.optimizer.step()
            # grad = self.optimizer.param_groups[0]['params'][0].grad
            # print(torch.mean(torch.abs(grad)))
            self.patch_obj.clamp_(p_min=self.min_epsilon, p_max=self.max_epsilon)
            self.reverse_step_size = self.optimizer.param_groups[0]['lr'] / 1000

    def attack_loss(self, confs):
        self.optimizer.zero_grad()
        loss = self.loss_fn(confs=confs, patch=self.detector_attacker.universal_patch[0])
        tv_loss, obj_loss = loss.values()
        tv_loss = torch.max(self.cfg.tv_eta * tv_loss, torch.tensor(0.1).to(self.device))
        # print(obj_loss)
        obj_loss = obj_loss * self.cfg.obj_eta
        loss = tv_loss.to(obj_loss.device) + obj_loss
        out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss}
        return out
