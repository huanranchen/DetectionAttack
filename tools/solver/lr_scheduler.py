from bisect import bisect_right
import torch
from torch import optim


class _ExponentStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma=0.999, step_size=1, last_epoch=1):
        self.gamma = gamma
        self.step_size = step_size
        super(_ExponentStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < 200 or self.last_epoch % self.step_size:
            return [group['lr'] for group in self.optimizer.param_groups]

        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]


class warmupALRS():
    """reference:Bootstrap Generalization Ability from Loss Landscape Perspective"""
    def __init__(self, optimizer, warmup_epoch=50, loss_threshold=1e-4, loss_ratio_threshold=1e-4, decay_rate=0.97):
        self.optimizer = optimizer

        self.warmup_rate = 1/3
        self.warmup_epoch = warmup_epoch
        self.start_lr = optimizer.param_groups[0]["lr"]
        self.warmup_lr = self.start_lr * (1-self.warmup_rate)
        self.update_lr(lambda x: x*self.warmup_rate)

        self.loss_threshold = loss_threshold
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold

        self.last_loss = 999

    def update_lr(self, update_fn):
        for group in self.optimizer.param_groups:
            group['lr'] = update_fn(group['lr'])
            now_lr = group['lr']
            print(f'now lr = {now_lr}')


    def step(self, loss, epoch):
        delta = self.last_loss - loss
        self.last_loss = loss
        if epoch < self.warmup_epoch:
            self.update_lr(lambda x: -(self.warmup_epoch-epoch)*self.warmup_lr / self.warmup_epoch + self.start_lr)
        elif delta < self.loss_threshold and delta/self.last_loss < self.loss_ratio_threshold:
            self.update_lr(lambda x: x*self.decay_rate)


class ALRS():
    """ALRS is a scheduler without warmup, a variant of warmupALRS."""
    def __init__(self, optimizer, loss_threshold=1e-4, loss_ratio_threshold=1e-4, decay_rate=0.97):
        self.optimizer = optimizer

        self.loss_threshold = loss_threshold
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold

        self.last_loss = 999

    def step(self, loss):
        delta = self.last_loss - loss
        if delta < self.loss_threshold and delta/self.last_loss < self.loss_ratio_threshold:
            for group in self.optimizer.param_groups:
                group['lr'] *= self.decay_rate
                now_lr = group['lr']
                print(f'now lr = {now_lr}')

        self.last_loss = loss


class ALRS_LowerTV():
    """
    A variant of the standard ALRS.
    This is just for observational scheduler comparison of the optimization to the Plateau_LR
        employed in the current baseline <Fooling automated surveillance cameras: adversarial patches to attack person detection>.
    The difference is that we fine-tune the hyper-params decay_rate from 0.97 to 0.955
        to force the learning rate down to 0.1 so that the TV Loss will converges to the same level.
    """
    def __init__(self, optimizer, loss_threshold=1e-4, loss_ratio_threshold=1e-4, decay_rate=0.955):
        self.optimizer = optimizer

        self.loss_threshold = loss_threshold
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold

        self.last_loss = 999

    def step(self, loss):
        delta = self.last_loss - loss
        if delta < self.loss_threshold and delta / self.last_loss < self.loss_ratio_threshold:
            for group in self.optimizer.param_groups:
                group['lr'] *= self.decay_rate
                now_lr = group['lr']
                print(f'now lr = {now_lr}')

        self.last_loss = loss


def Expo_lr_scheduler(optimizer):
    return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999, update_step=5)


def Cosine_lr_scheduler(optimizer, total_epoch=1000):
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch)


def Plateau_lr_scheduler(optimizer, patience=100):
    return optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience)