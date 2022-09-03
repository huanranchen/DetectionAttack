import numpy as np
import torch
from .base import BaseAttacker


class OptimAttacker(BaseAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def patch_update(self, **kwargs):
        patch_clamp_ = kwargs['patch_clamp_']
        self.optimizer.step()
        patch_clamp_(p_min=self.min_epsilon, p_max=self.max_epsilon)

    def attack_loss(self, confs):
        eta = 1
        if hasattr(self.cfg, 'tv_eta'):
            eta = self.cfg.tv_eta
        self.optimizer.zero_grad()

        loss = self.loss_fn(confs=confs, patch=self.detector_attacker.universal_patch[0])

        if self.cfg.LOSS_FUNC == 'obj-tv':
            tv_loss, obj_loss = loss.values()
            tv_loss = torch.max(eta * tv_loss, torch.tensor(0.1).to(self.device))
            obj_loss = obj_loss * self.cfg.obj_eta
            loss = tv_loss + obj_loss
            if self.detector_attacker.vlogger:
                self.detector_attacker.vlogger.write_loss(loss, obj_loss, tv_loss)
            # print('tv loss: ', tv_loss, 'obj loss:', obj_loss)

        elif self.cfg.LOSS_FUNC == 'obj':
            loss = loss['obj_loss'] * self.cfg.obj_eta
            if self.detector_attacker.vlogger:
                self.detector_attacker.vlogger.write_scalar(loss, 'loss/det_loss')
        return loss