import matplotlib.pyplot as plt
from .optim import OptimAttacker
import torch
import os
import numpy as np
import cv2
import random
import datetime


def get_datetime_str(style='dt'):
    cur_time = datetime.datetime.now()
    date_str = cur_time.strftime('%y_%m_%d_')
    time_str = cur_time.strftime('%H_%M_%S')
    if style == 'data':
        return date_str
    elif style == 'time':
        return time_str
    return date_str + time_str


class RSCchr(OptimAttacker):
    '''
    思路从RSC获得，但通过解空间理论做了改进
    mask掉k%的梯度大的区域，去训练。去强化新特征
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def non_targeted_attack(self, ori_tensor_batch, detector, mask_prob=0.1):
        losses = []
        assert self.iter_step == 2, 'if you use RSCchr, should iterate only twice'
        for iteration in range(1, self.iter_step + 1):
            if iteration > 0:
                ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            # detect adv img batch to get bbox and obj confs
            bboxes, confs, cls_array = detector(adv_tensor_batch).values()

            if hasattr(self.cfg, 'class_specify'):
                attack_cls = int(self.cfg.ATTACK_CLASS)
                confs = torch.cat(
                    ([conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
            elif hasattr(self.cfg, 'topx_conf'):
                confs = torch.sort(confs, dim=-1, descending=True)[0][:, :self.cfg.topx_conf]
                confs = torch.mean(confs, dim=-1)
            else:
                confs = confs.max(dim=-1, keepdim=True)[0]

            detector.zero_grad()
            loss_dict = self.attack_loss(confs=confs)
            loss = loss_dict['loss']
            loss.backward()

            # # do the visualizations
            # if random.random() < 0.1:
            #     patch = self.optimizer.param_groups[0]['params'][0]
            #     self.visualize_patch_saliency_map(patch.detach(), patch.grad)

            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update(patch_clamp_=self.detector_attacker.patch_obj.clamp_)

            # do the mask
            if iteration == 1:
                patch = self.optimizer.param_groups[0]['params'][0]
                original_patch = patch.clone().detach()  # backup
                grad = patch.grad
                mask = self.get_mask(self.get_patched_activation(grad))
                with torch.no_grad():
                    patch.mul_(~mask)
            if iteration == 2:
                patch = self.optimizer.param_groups[0]['params'][0]
                with torch.no_grad():
                    patch[mask] = original_patch[mask]  # restore the patch

        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

    def get_mask(self, x: torch.tensor, mask_ratio=0.25):
        sorted_x = torch.sort(x.reshape(-1), dim=0, descending=True)[0]
        threshold = sorted_x[int(sorted_x.shape[0] * mask_ratio)]
        mask = x >= threshold
        return mask

    @staticmethod
    def get_patched_activation(x: torch.tensor, small_patch_size: tuple = (20, 20)):
        '''
        把图片分成若干个小patch，每个patch部分求均值
        '''
        shape = x.shape
        x = x.squeeze()  # 3, 300, 300
        height, width = x.shape[1], x.shape[2]
        num_patch_height = height // small_patch_size[0]
        num_patch_width = width // small_patch_size[1]
        x = x.reshape(3, num_patch_height, small_patch_size[0],
                      num_patch_width, small_patch_size[1])
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2],
                      small_patch_size[0] * small_patch_size[1])
        x = torch.sum(torch.abs(x), dim=-1, keepdim=True)  # 此处是否加abs存疑
        x = x.repeat(1, 1, 1, small_patch_size[0] * small_patch_size[1])
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2],
                      small_patch_size[0], small_patch_size[1])
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(3, height, width)

        # handle the channel
        x = torch.sum(x, dim=0, keepdim=True)
        x = x.repeat(3, 1, 1)

        return x.reshape(shape)

    @torch.no_grad()
    def visualize_patch_saliency_map(self, patch_data: torch.tensor, patch_grad: torch.tensor,
                                     save_path: str = './patch_saliency/',
                                     small_patch_size: tuple = (20, 20)):
        '''
        patch
        a patch is worth 15x15 patch!
        '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        patch_data = self.tensor2cv2image(patch_data)

        # process grad, parallel computing:
        patch_grad = self.get_patched_activation(patch_grad, small_patch_size)
        patch_grad = self.normalize(patch_grad)  # 3, 300, 300

        patch_grad = self.tensor2cv2image(patch_grad)

        height, width, _ = patch_data.shape  # 读取输入图片的尺寸

        heatmap = cv2.applyColorMap(patch_grad, cv2.COLORMAP_JET)  # get heat map
        result = heatmap * 0.5 + patch_data * 0.5  # 比例可以自己调节

        cv2.imwrite(os.path.join(save_path, get_datetime_str() + '.jpg'), result)

    @staticmethod
    def normalize(x: torch.tensor) -> torch.tensor:
        '''
        because grad is so small, so we had better to normalize it
        '''
        assert torch.min(x).item() >= 0, 'the smallest value should bigger than 0'
        maximum = torch.max(x).item()
        scale = 1 / maximum
        x *= scale
        print(torch.max(x), torch.mean(x))
        return x

    @staticmethod
    def tensor2cv2image(x: torch.tensor) -> np.array:
        x = x.squeeze()
        x = x.permute(1, 2, 0)
        x = x.cpu().numpy()
        x *= 255
        x = np.uint8(x)
        return x


class StrengthenWeakPointAttacker(OptimAttacker):
    """
    找出共同弱点的部分，单独进行k次攻击，强化共同弱点
    """

    def __init__(self, *args, **kwargs):
        # 因为这里的args的顺序和继承的OptimAttacker不一样，所以会不会有错？做个实验！
        super().__init__(*args, **kwargs)

    def non_targeted_attack(self, ori_tensor_batch, detector):
        self.ori_tensor_batch = ori_tensor_batch
        self.detectors.append(detector)
        losses = []
        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            adv_tensor_batch = adv_tensor_batch.to(detector.device)
            # detect adv img batch to get bbox and obj confs
            bboxes, confs, cls_array = detector(adv_tensor_batch).values()

            if hasattr(self.cfg, 'class_specify'):
                attack_cls = int(self.cfg.ATTACK_CLASS)
                confs = torch.cat(
                    ([conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
            elif hasattr(self.cfg, 'topx_conf'):
                # attack top x confidence
                # print(confs.size())
                confs = torch.sort(confs, dim=-1, descending=True)[0][:, :self.cfg.topx_conf]
                confs = torch.mean(confs, dim=-1)
            else:
                # only attack the max confidence
                confs = confs.max(dim=-1, keepdim=True)[0]

            detector.zero_grad()
            # print('confs', confs)
            loss_dict = self.attack_loss(confs=confs)
            loss = loss_dict['loss']
            # print(loss)
            loss.backward()
            self.grad_record.append(self.optimizer.param_groups[0]['params'][0].grad.clone())
            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update()
        # print(adv_tensor_batch, bboxes, loss_dict)
        # update training statistics on tensorboard
        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

    @staticmethod
    def get_topk_mask(x: torch.tensor, mask_ratio=0.5):
        sorted_x = torch.sort(x.reshape(-1), dim=0, descending=True)[0]
        threshold = sorted_x[int(sorted_x.shape[0] * mask_ratio)]
        mask = x >= threshold
        return mask

    @torch.no_grad()
    def begin_attack(self):
        # self.original_patch = self.optimizer.param_groups[0]['params'][0].detach().clone()
        self.grad_record = []
        self.detectors = []

    @staticmethod
    def IOU_for_mask(x: list):
        intersection = torch.ones_like(x[0].to(torch.float))
        union = torch.zeros_like(x[0].to(torch.float))
        for now in x:
            intersection *= now
            union += now
        union[union < 1] = 0
        union[union >= 1] = 1
        iou = torch.sum(intersection) / torch.sum(union)
        return iou

    def end_attack(self, strengthen_time=1):
        self.masks = []
        for grad in self.grad_record:
            self.masks.append(self.get_topk_mask(torch.abs(grad)))  # attention! 加了abs！！！！！

        # record grad_similarity
        grad_similarity = self.IOU_for_mask(self.masks)
        self.detector_attacker.vlogger.write_scalar(grad_similarity, 'grad_similarity')

        # find common weakness
        mask = torch.ones_like(self.masks[0].to(torch.float))
        for now_mask in self.masks:
            mask *= now_mask

        # forward and strengthen
        for i in range(strengthen_time):
            self.non_targeted_attack(self.ori_tensor_batch, self.detectors[i])

        # delete cache
        del self.grad_record
        del self.detectors
