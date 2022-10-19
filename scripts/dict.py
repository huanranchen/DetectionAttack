import torch
from tools.solver import Cosine_lr_scheduler, Plateau_lr_scheduler, ALRS, warmupALRS
from attack.methods import LinfBIMAttack, LinfMIMAttack, LinfPGDAttack, OptimAttacker, \
    FishAttacker, SmoothFishAttacker, OptimAttackerWithRecord, RSCchr, StrengthenWeakPointAttacker
from tools.solver.loss import *

scheduler_factory = {
    'plateau': Plateau_lr_scheduler,
    'cosine': Cosine_lr_scheduler,
    'ALRS': ALRS,
    'warmupALRS': warmupALRS
}

optim_factory = {
    'optim': lambda p_obj, lr: torch.optim.Adam([p_obj], lr=lr, amsgrad=True),  # default
    'optim-adam': lambda p_obj, lr: torch.optim.Adam([p_obj], lr=lr, amsgrad=True),
    'optim-sgd': lambda p_obj, lr: torch.optim.SGD([p_obj], lr=lr * 100),
    'optim-nesterov': lambda p_obj, lr: torch.optim.SGD([p_obj], lr=lr * 100, nesterov=True, momentum=0.9),
    'optim-rmsprop': lambda p_obj, lr: torch.optim.RMSprop([p_obj], lr=lr * 100),
    "IDGM-fish": lambda p_obj, lr: torch.optim.Adam([p_obj], lr=lr, amsgrad=True),  # default
    'optim-record': lambda p_obj, lr: torch.optim.Adam([p_obj], lr=lr, amsgrad=True),  # default
    'IDGM-smoothfish': lambda p_obj, lr: torch.optim.Adam([p_obj], lr=lr, amsgrad=True),  # default
}

attack_method_dict = {
    "": None,
    "bim": LinfBIMAttack,
    "mim": LinfMIMAttack,
    "pgd": LinfPGDAttack,
    "optim": OptimAttacker,
    "IDGM-fish": FishAttacker,
    "IDGM-smoothfish": SmoothFishAttacker,
    "optim-record": OptimAttackerWithRecord,
}

loss_dict = {
    '': None,
    "ascend-mse": ascend_mse_loss,  # for gradient sign-based method
    "descend-mse": descend_mse_loss,  # for gradient sign-based method
    "obj-tv": obj_tv_loss,  # for optim(MSE as well)
}


def get_attack_method(attack_method: str):
    if 'optim' in attack_method:
        return attack_method_dict['optim']
    return attack_method_dict[attack_method]
