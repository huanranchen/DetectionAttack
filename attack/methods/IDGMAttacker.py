import torch
from .optim import OptimAttacker


class FishAttacker(OptimAttacker):
    '''
    use fish technique to approximate second derivatives.
    '''

    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(device, cfg, loss_func, detector_attacker, norm=norm)

    def begin_attack(self):
        self.original_patch = self.optimizer.param_groups[0]['params'][0].detach().clone()

    def end_attack(self, ksi=0.1):
        '''
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        '''
        patch = self.optimizer.param_groups[0]['params'][0]
        patch.mul_(ksi)
        patch.add_((1 - ksi) * self.original_patch)
        self.original_patch = None
