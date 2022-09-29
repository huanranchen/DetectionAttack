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
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []
        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            # detect adv img batch to get bbox and obj confs
            bboxes, confs, cls_array = detector(adv_tensor_batch).values()

            if hasattr(self.cfg, 'class_specify'):
                attack_cls = int(self.cfg.ATTACK_CLASS)
                confs = torch.cat(
                    ([conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
            elif hasattr(self.cfg, 'topx_conf'):
                # attack top x confidence
                confs = torch.sort(confs, dim=-1, descending=True)[0][:, :self.cfg.topx_conf]
                confs = torch.mean(confs, dim=-1)
            else:
                # only attack the max confidence
                confs = confs.max(dim=-1, keepdim=True)[0]

            detector.zero_grad()
            loss_dict = self.attack_loss(confs=confs)
            loss = loss_dict['loss']
            loss.backward()

            # do the visualizations
            if random.random() < 0.1:
                patch = self.optimizer.param_groups[0]['params'][0]
                self.visualize_patch_saliency_map(patch.detach(), patch.grad)

            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update(patch_clamp_=self.detector_attacker.patch_obj.clamp_)

        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

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

        print(patch_grad.shape)
        # process grad, parallel computing:
        patch_grad = patch_grad.squeeze()  # 3, 300, 300
        height, width = patch_grad.shape[1], patch_grad.shape[2]
        num_patch_height = height // small_patch_size[0]
        num_patch_width = width // small_patch_size[1]
        patch_grad = patch_grad.reshape(3, num_patch_height, small_patch_size[0], num_patch_width, small_patch_size[1])
        patch_grad = patch_grad.permute(0, 2, 4, 1, 3)
        patch_grad = patch_grad.reshape(patch_grad.shape[0], patch_grad.shape[1], patch_grad.shape[2],
                                        num_patch_height * num_patch_width)
        patch_grad = torch.sum(torch.abs(patch_grad), dim=-1, keepdim=True)  # 此处是否加abs存疑
        patch_grad = patch_grad.repeat(1, 1, 1, num_patch_width * num_patch_height)
        patch_grad = patch_grad.reshape(patch_grad.shape[0], patch_grad.shape[1], patch_grad.shape[2],
                                        num_patch_height, num_patch_width)
        patch_grad = patch_grad.permute(0, 3, 1, 4, 2)
        patch_grad = patch_grad.reshape(3, height, width)

        patch_grad = self.normalize(patch_grad)
        ################################

        patch_grad = self.tensor2cv2image(patch_grad)

        height, width, _ = patch_data.shape  # 读取输入图片的尺寸

        heatmap = cv2.applyColorMap(patch_grad, cv2.COLORMAP_JET)  # get heat map
        result = heatmap * 0.5 + patch_data * 0.5  # 比例可以自己调节

        cv2.imwrite(os.path.join(save_path, get_datetime_str() + '.jpg'), result)
        plt.imshow(heatmap)
        plt.savefig(os.path.join(save_path, get_datetime_str() + '.jpg'))
        plt.close()

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
