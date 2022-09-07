import torch
from abc import ABC, abstractmethod


class BaseAttacker(ABC):
    """An Attack Base Class"""

    def __init__(self, loss_func, norm: str, cfg, device: torch.device, detector_attacker):
        """

        :param loss_func:
        :param norm: str, [L0, L1, L2, L_infty]
        :param cfg:
        :param detector_attacker: this attacker should have attributes vlogger

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
        self.max_iters = cfg.MAX_EPOCH
        self.iter_step = cfg.ITER_STEP
        self.step_size = cfg.STEP_SIZE
        self.class_id = cfg.TARGET_CLASS
        self.attack_class = cfg.ATTACK_CLASS

    def logger(self, detector, adv_tensor_batch, bboxes, loss_dict):
        vlogger = self.detector_attacker.vlogger
        if vlogger and vlogger.iter % 500:
            filter_box = self.detector_attacker.filter_bbox
            vlogger.write_tensor(self.detector_attacker.universal_patch[0], 'adv patch')
            vlogger.write_loss(loss_dict['loss'], loss_dict['det_loss'], loss_dict['tv_loss'])
            plotted = self.detector_attacker.plot_boxes(adv_tensor_batch[0], filter_box(bboxes[0]))
            vlogger.write_cv2(plotted, f'{detector.name}')

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []
        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            # detect adv img batch to get bbox and obj confs
            bboxes, confs, cls_array = detector(adv_tensor_batch).values()

            if hasattr(self.cfg, 'class_specify'):
                # TODO: only support filtering a single cls now
                attack_cls = int(self.cfg.ATTACK_CLASS)
                confs = torch.cat(([conf[cls==attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
            else:
                # only attack the max confidence
                confs = confs.max(dim=-1, keepdim=True)[0]

            detector.zero_grad()
            loss_dict = self.attack_loss(confs=confs)
            loss = loss_dict['loss']
            loss.backward()
            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update(patch_clamp_=self.detector_attacker.patch_obj.clamp_)

        # update training statistics on tensorboard
        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

    @abstractmethod
    def patch_update(self, **kwargs):
        pass

    @abstractmethod
    def attack_loss(self, confs):
        pass