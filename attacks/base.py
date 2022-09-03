import torch
import numpy as np
from abc import ABC, abstractmethod


class BaseAttacker(ABC):
    """An Attack Base Class"""
    def __init__(self, loss_func, norm:str, cfg, device: torch.device, detector_attacker):
        """

        :param loss_func:
        :param norm: str, [L0, L1, L2, L_infty]
        :param cfg:

        Args:
            loss_func ([torch.nn.Loss]): [a loss function to calculate the loss between the inputs and expeced outputs]
            norm (str, optional): [the attack norm and the choices are [L0, L1, L2, L_infty]]. Defaults to 'L_infty'.
            epsilons (float, optional): [the upper bound of perturbation]. Defaults to 0.05.
            max_iters (int, optional): [the maximum iteration number]. Defaults to 10.
            step_size (float, optional): [the step size of attack]. Defaults to 0.01.
            device ([type], optional): ['cpu' or 'cuda']. Defaults to None.
        """
        self.loss_fn = loss_func
        self.cfg = cfg
        self.detector_attacker = detector_attacker
        self.device = device
        self.norm = norm
        self.min_epsilon = 0.
        self.max_epsilon = cfg.EPSILON / 255.
        self.max_iters = cfg.MAX_ITERS
        self.iter_step = cfg.ITER_STEP
        self.step_size = cfg.STEP_SIZE
        self.class_id = cfg.TARGET_CLASS
        self.attack_class = cfg.ATTACK_CLASS

    def logger(self, detector, adv_tensor_batch, bboxes):
        if self.detector_attacker.vlogger and self.detector_attacker.vlogger.iter % 5:
            self.detector_attacker.vlogger.write_tensor(self.detector_attacker.universal_patch[0], 'adv patch')
            plotted = self.detector_attacker.plot_boxes(adv_tensor_batch[0], bboxes[0])
            self.detector_attacker.vlogger.write_cv2(plotted, f'{detector.name}')

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []

        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            # detect adv img batch to get bbox and obj confs
            bboxes, confs, cls_array = detector(adv_tensor_batch).values()

            if hasattr(self.cfg, 'class_specify'):
                filter_box = self.detector_attacker.filter_bbox
                confs = torch.cat(([filter_box(conf, cls).max(dim=-1, keepdim=True)[0]
                                for conf, cls in zip(confs, cls_array)]))
            else: confs = confs.max(dim=-1, keepdim=True)[0]

            detector.zero_grad()
            loss = self.attack_loss(confs)
            loss.backward()
            losses.append(float(loss))

            self.patch_update(patch_clamp_=self.detector_attacker.patch_obj.clamp_)

        self.logger(detector, adv_tensor_batch, bboxes)
        return torch.tensor(losses).mean()

    def total_loss(self, det_loss):
        pass

    @abstractmethod
    def patch_update(self, **kwargs):
        pass

    @abstractmethod
    def attack_loss(self, confs):
        pass