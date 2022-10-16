from tools.solver import Cosine_lr_scheduler, Plateau_lr_scheduler, ALRS, warmupALRS

scheduler_factory = {
    'plateau': Plateau_lr_scheduler,
    'cosine': Cosine_lr_scheduler,
    'ALRS': ALRS,
    'warmupALRS': warmupALRS
}

import torch

optim_factory = {
    'optim': lambda p_obj, lr: torch.optim.Adam([p_obj], lr=lr, amsgrad=True),  # default
    'optim-adam': lambda p_obj, lr: torch.optim.Adam([p_obj], lr=lr, amsgrad=True),
    'optim-sgd': lambda p_obj, lr: torch.optim.SGD([p_obj], lr=lr * 100),
    'optim-nesterov': lambda p_obj, lr: torch.optim.SGD([p_obj], lr=lr * 100, nesterov=True, momentum=0.9),
    'optim-rmsprop': lambda p_obj, lr: torch.optim.RMSprop([p_obj], lr=lr)
}
