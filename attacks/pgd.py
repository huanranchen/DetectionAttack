import copy
import sys

from .base import Base_Attacker

import torch
import torch.distributed as dist
import numpy as np

num_iter = 0
update_pre = 0
# patch_tmp = None

class LinfPGDAttack(Base_Attacker):
    """
        PGD attacks
        epsilon: magnitude of attack
        k: iterations
        a: step size
    """

    def __init__(self, loss_fuction, model, norm='L_infty', epsilons=0.05, max_iters=50, step_size=0.01, class_id=0, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), momentum=0.5):
        """this is init function of PGD attack (arxiv: https://arxiv.org/pdf/1706.06083.pdf)

        Args:
            loss_fuction ([torch.nn.Loss]): [a loss function to calculate the loss between the inputs and expeced outputs]
            model ([torch.nn.model]): [target model to attack].
            norm (str, optional): [the attack norm and the choices are [L0, L1, L2, L_infty]]. Defaults to 'L_infty'.
            epsilons (float, optional): [the upper bound of perturbation]. Defaults to 0.05.
            max_iters (int, optional): [the maximum iteration number]. Defaults to 10.
            step_size (float, optional): [the step size of attack]. Defaults to 0.01.
            device ([type], optional): ['cpu' or 'cuda']. Defaults to None.
            momentum (float, optional): [the momentum of attack]. Defaults to 0.5.
        """
        super(LinfPGDAttack, self).__init__(model, norm, epsilons, None)
        self.max_iters = max_iters
        self.step_size = step_size
        self.device = device
        self.momentum = momentum
        self.loss_fn = loss_fuction
        self.epsilon = epsilons
        self.class_id = class_id
        # self.eta = torch.nn.Parameter(torch.ones((1, 3, 128, 128), requires_grad=True).cuda())
        # self.optim = torch.optim.Adam([{'params': self.eta}], lr=0.01)
     
    def non_targeted_attack(self, ori_img_tensor, adv_img_tensor, img_cv2, detector_attacker, detector):

        c1 = [0, 0, 0]
        c2 = [self.epsilon, self.epsilon, self.epsilon]
        c = np.vstack([c1,c2])
        c = np.resize(c, (1, 2, 3))
        epsilon = detector.normalize(c)
        
        min_epsilon = [epsilon[0, 0, 0, 0].item(), epsilon[0, 1, 0, 0].item(), epsilon[0, 2, 0, 0].item()]
        max_epsilon = [epsilon[0, 1, 0, 1].item(), epsilon[0, 1, 0, 1].item(), epsilon[0, 2, 0, 1].item()]
        # print(min_epsilon, max_epsilon)

        for iter in range(self.max_iters):
            preds, detections_with_grad = detector.detect_img_tensor_get_bbox_conf(input_img=adv_img_tensor, ori_img_cv2=img_cv2)
            disappear_loss = self.loss_fn(detections_with_grad)
            detector.zero_grad()
            disappear_loss.backward(retain_graph=True)

            for i in range(len(detector_attacker.patch_boxes)):
                grad = detector_attacker.patch_boxes[i][-1].grad
                patch = detector_attacker.patch_boxes[i][-1]
                # print('leaf', patch.is_leaf, patch.requires_grad)
                patch = patch + self.step_size * grad.sign()
                patch[0,0,:,:] = torch.clamp(patch[0,0,:,:], min=min_epsilon[0], max=max_epsilon[0])
                patch[0,1,:,:] = torch.clamp(patch[0,1,:,:], min=min_epsilon[1], max=max_epsilon[1])
                patch[0,2,:,:] = torch.clamp(patch[0,2,:,:], min=min_epsilon[2], max=max_epsilon[2])
                detector_attacker.patch_boxes[i][-1] = patch
                
            detector.zero_grad()
            adv_img_tensor = detector_attacker.apply_patches(ori_img_tensor, detector, is_normalize=False)
            avg_conf = sum(item[4] for item in preds) / (len(preds) + 0.01)
            # print(len(preds), avg_conf, disappear_loss.item())
        return adv_img_tensor

    def init_epsilon(self, detector):
        c1 = [0, 0, 0]
        c2 = [self.epsilon, self.epsilon, self.epsilon]
        c = np.vstack([c1, c2])
        c = np.resize(c, (1, 2, 3))
        # print("init: ", c.shape, c.dtype)
        epsilon = detector.normalize(c)

        self.min_epsilon = [epsilon[0, 0, 0, 0].item(), epsilon[0, 1, 0, 0].item(), epsilon[0, 2, 0, 0].item()]
        self.max_epsilon = [epsilon[0, 1, 0, 1].item(), epsilon[0, 1, 0, 1].item(), epsilon[0, 2, 0, 1].item()]

    def clamp(self, im_tensor, max_epsilon, min_epsilon):
        im_tensor[:, 0, :, :] = torch.clamp(im_tensor[:, 0, :, :], min=min_epsilon[0], max=max_epsilon[0])
        im_tensor[:, 1, :, :] = torch.clamp(im_tensor[:, 1, :, :], min=min_epsilon[1], max=max_epsilon[1])
        im_tensor[:, 2, :, :] = torch.clamp(im_tensor[:, 2, :, :], min=min_epsilon[2], max=max_epsilon[2])
        return im_tensor

    def serial_non_targeted_attack(self, ori_tensor_batch, detector_attacker, detector, confs_thresh=0.3):
        global update_pre, num_iter
        adv_tensor_batch, patch_tmp = detector_attacker.apply_universal_patch(ori_tensor_batch, detector)
        print('conf thresh: ', confs_thresh)
        # interative attack
        # self.optim.zero_grad()
        losses = []
        for iter in range(detector_attacker.cfg.ATTACKER.SERIAL_ITER_STEP):
            num_iter += 1

            # detect adv img batch to get bbox and obj confs
            preds, detections_with_grad = detector.detect_img_batch_get_bbox_conf(
                adv_tensor_batch, confs_thresh=confs_thresh)
            bbox_num = torch.FloatTensor([len(pred) for pred in preds])
            # print('bbox num: ', bbox_num)
            if torch.sum(bbox_num) == 0: break
            detector.zero_grad()

            if hasattr(detector_attacker.cfg.DETECTOR, 'ETA') and detector_attacker.cfg.DETECTOR.ETA:
                disappear_loss, update_func = self.loss_fn(detections_with_grad * bbox_num)
            else:
                disappear_loss, update_func = self.loss_fn(detections_with_grad)
                # print("in pgd: ", detections_with_grad[:2])
            disappear_loss.backward()
            grad = patch_tmp.grad

            if hasattr(detector_attacker.cfg.ATTACKER, 'nesterov'):
                momentum_step = detector_attacker.cfg.ATTACKER.nesterov
                # momentum初期修正
                # if num_iter < 100:
                #     update /= (1 - momentum_step)
                # patch_tmp = patch_tmp - momentum_step * update * self.step_size
                update = self.step_size * grad
                l2 = torch.sqrt(torch.sum(torch.pow(update, 2))) / update.numel()
                print(l2)
                update /= l2
                # update = momentum_step * update_pre + self.step_size * grad
                # update_pre = copy.deepcopy(update)
            else:
                update = self.step_size * grad.sign()

            patch_tmp = update_func(patch_tmp, update)
            losses.append(float(disappear_loss))
            # min_max epsilon clamp of different channels
            patch_tmp = self.clamp(patch_tmp, self.max_epsilon, self.min_epsilon)
            adv_tensor_batch, _ = detector_attacker.apply_universal_patch(
                ori_tensor_batch, detector, is_normalize=False, universal_patch=patch_tmp)
        # if hasattr(detector_attacker.cfg.DETECTOR, 'ETA') and detector_attacker.cfg.DETECTOR.ETA:
        #     self.optim.step()
        patch_tmp = detector.unnormalize_tensor(patch_tmp.detach())
        # sys.exit()
        # print(losses)
        # print(' Loss: ', np.mean(losses))
        return patch_tmp, np.mean(losses)

    def parallel_non_targeted_attack(self, ori_tensor_batch, detector_attacker, detector):
        adv_tensor_batch, patch_tmp = detector_attacker.apply_universal_patch(ori_tensor_batch, detector)

        # max_epsilon, min_epsilon = self.init_epsilon(detector)
        for iter in range(detector_attacker.cfg.ATTACKER.PARALLEL_ITER_STEP):
            preds, detections_with_grad = detector.detect_img_batch_get_bbox_conf(adv_tensor_batch)
            disappear_loss = self.loss_fn(detections_with_grad)

            if detector_attacker.ddp:

                print('Rank ',
                      dist.get_rank(), disappear_loss)
                dist.all_reduce(disappear_loss, op=dist.ReduceOp.AVG)

            # print('leaf1: ', patch_tmp)
            detector.zero_grad()
            disappear_loss.backward()

            grad = patch_tmp.grad
            patch_tmp = patch_tmp + self.step_size * grad.sign()
            # print('get grad!-----------------------------')
            patch_tmp = self.clamp(patch_tmp, self.max_epsilon, self.min_epsilon)

            adv_tensor_batch, _ = detector_attacker.apply_universal_patch(ori_tensor_batch, detector,
                                                                     is_normalize=False, universal_patch=patch_tmp)

            # detector_attacker.save_patch('./results/self'+str(iter)+'.png', patch=patch_tmp)
            # detector_attacker.save_patch('./results/tmp'+str(iter)+'.png', patch=detector_attacker.universal_patch)
            # print('is equal: ', torch.equal(patch_tmp, detector_attacker.universal_patch))
        patch_tmp = detector.unnormalize_tensor(patch_tmp.detach())
        patch_updates = patch_tmp - detector_attacker.universal_patch

        return patch_updates
            