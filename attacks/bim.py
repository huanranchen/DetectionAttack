from .base import Base_Attacker

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import cv2
import copy
from tqdm import tqdm

class LinfBIMAttack(Base_Attacker):
    """
        BIM attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
    """

    def __init__(self, loss_fuction, model, norm='L_infty', epsilons=0.05, max_iters=10, step_size=0.01, perturbation=None, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """this is init function of BIM attack (arxiv: https://arxiv.org/pdf/1607.02533.pdf)

        Args:
            loss_fuction ([torch.nn.Loss]): [a loss function to calculate the loss between the inputs and expeced outputs]
            model ([torch.nn.model]): [target model to attack].
            norm (str, optional): [the attack norm and the choices are [L0, L1, L2, L_infty]]. Defaults to 'L_infty'.
            epsilons (float, optional): [the upper bound of perturbation]. Defaults to 0.05.
            max_iters (int, optional): [the maximum iteration number]. Defaults to 10.
            step_size (float, optional): [the step size of attack]. Defaults to 0.01.
            device ([type], optional): ['cpu' or 'cuda']. Defaults to None.
        """
        super(LinfBIMAttack, self).__init__(model, norm, epsilons, perturbation)
        self.max_iters = max_iters
        self.step_size = step_size
        self.device = device
        self.loss_fn = loss_fuction
        self.epsilon = epsilons
     
    def non_targeted_attack(self, x, y, is_universal = False, x_min=-10, x_max=10, *model_args):
        """the main attack method of BIM

        Args:
            x ([torch.tensor]): [input of model]
            y ([torch.tensor]): [expected or unexpected outputs]
            x_min (int, optional): [the minimum value of input x]. Defaults to -1.
            x_max (int, optional): [the maximum value of input x]. Defaults to 1.

        Returns:
            [tensor]: [the adversarial examples crafted by BIM]
        """
        x_ori = x.clone().detach_()
        x_adv = x.clone().detach_()
        for iter in range(self.max_iters):
            x_adv.requires_grad = True
            output = self.model(x_adv, model_args)
            self.model.zero_grad()
            loss = self.loss_fn(output, y)
            loss.backward()
            grad = x_adv.grad
            X_adv = x_adv + self.step_size * grad.sign()
            eta = torch.clamp(X_adv - x_ori, min=-self.epsilon, max=self.epsilon)
            if is_universal:
                eta = torch.mean(eta.detach_(),dim=0)
            x_adv = torch.clamp(x_ori + eta, min=x_min, max=x_max).detach_()
        self.model.zero_grad()
        self.perturbation = eta
        return x_adv
    
    def non_targeted_attack_mmdet(self, data, img_cv2, img, img_resized, test_pipeline, prase_func, is_prase=True, is_universal = False, x_min=-255, x_max=255):
        """the main attack method of BIM

        Args:
            data ([torch.tensor]): [input of model]
            img_cv2 ([np.array]): [image readed by opencv-python]
            img ([torch.tensor]): [tensor transformed from img_cv2]
            img_resized ([torch.tensor]): [resized tensor transformed from img_cv2]
            x_min (int, optional): [the minimum value of input x]. Defaults to -1.
            x_max (int, optional): [the maximum value of input x]. Defaults to 1.

        Returns:
            [tensor]: [the adversarial examples crafted by BIM]
        """
        x_ori = copy.deepcopy(img)
        x_adv = img
        x_adv.requires_grad = True
        original_boxes_number = -1
        pbar = tqdm(range(self.max_iters), desc='attacking image ......')
        for iter in pbar:
            results, cls_scores, det_bboxes = self.model(return_loss=False, rescale=True, **data)
            if is_prase:
                bboxes, labels, bbox_num = prase_func(self.model, img_cv2, results[0])
                if original_boxes_number==-1:
                    original_boxes_number = bbox_num
                pbar.set_description('attacking image... original boxes: {} / now boxes: {}'.format(original_boxes_number, bbox_num))
            self.model.zero_grad()
            loss = self.loss_fn(det_bboxes[0][:,-1])
            loss.backward()
            grad = x_adv.grad
            X_adv = x_adv + self.step_size * grad.sign()
            eta = torch.clamp(X_adv - x_ori, min=-self.epsilon, max=self.epsilon)
            if is_universal:
                eta = torch.mean(eta.detach_(),dim=0)
            x_adv = torch.clamp(x_ori + eta, min=x_min, max=x_max).detach_()
            x_adv.requires_grad = True
            data['img'] = x_adv
            data = test_pipeline(data)
        self.model.zero_grad()
        self.perturbation = eta
        x_adv = x_adv.cpu().detach().clone().squeeze(0).numpy().transpose(1, 2, 0)
        x_adv_cv2 = cv2.cvtColor(x_adv, cv2.COLOR_RGB2BGR)
        return x_adv, x_adv_cv2