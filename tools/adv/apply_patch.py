"""

This is not used since tons of tensors takes huge GPU memory
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .median_pool import MedianPool2d


class PatchTransformer(nn.Module):
    def __init__(self, device, rotate_angle=20, rand_shift_rate=0.4, scale_rate=0.2):
        """

        :param rotate_angle: random rotate angle range from [-rotate_angle, rotate_angle]
        :param rand_loc_rate: random shift rate (of patch) range
        :param scale_rate: patch size / bbox size
        """
        super().__init__()
        self.min_rotate_angle = -rotate_angle / 180 * math.pi
        self.max_rotate_angle = rotate_angle / 180 * math.pi
        self.rand_shift_rate = rand_shift_rate
        self.scale_rate = scale_rate
        self.median_pooler = MedianPool2d(7, same=True)
        self.device = device
        # self.max_n_labels = 10

    def random_shift(self, x, limited_range):
        shift = limited_range * torch.FloatTensor(x.size()).uniform_(-self.rand_shift_rate, self.rand_shift_rate)
        return x + shift

    def random_jitter(self, x, min_contrast=0.8, max_contrast=1.2, min_brightness=-0.1, max_brightness=0.1, noise_factor = 0.10):
        bboxes_shape = torch.Size((x.size(0), x.size(1)))
        contrast = torch.FloatTensor(bboxes_shape).uniform_(min_contrast, max_contrast)
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        contrast = contrast.expand(-1, -1, x.size(-3), x.size(-2), x.size(-1)).to(self.device)

        # Create random brightness tensor
        brightness = torch.FloatTensor(bboxes_shape).uniform_(min_brightness, max_brightness)
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        brightness = brightness.expand(-1, -1, x.size(-3), x.size(-2), x.size(-1)).to(self.device)

        # Create random noise tensor
        noise = torch.FloatTensor(x.size()).uniform_(-1, 1) * noise_factor

        # Apply contrast/brightness/noise, clamp
        x = contrast * x + brightness + noise.to(self.device)
        x = torch.clamp(x, 0.000001, 0.99999)
        return x

    def forward(self, adv_patch_batch, bboxes_batch, patch_ratio, rand_rotate_gate=True, rand_shift_gate=False):
        batch_size = bboxes_batch.size(0)
        lab_len = bboxes_batch.size(1)
        bboxes_size = np.prod([batch_size, lab_len])
        # print(bboxes_batch[0][:4, :])

        # -------------Shift & Random relocate--------------
        # bbox format is [x1, y1, x2, y2, conf, cls_id]
        bw = torch.clamp(bboxes_batch[:, :, 2] - bboxes_batch[:, :, 0], min=0, max=1)
        bh = torch.clamp(bboxes_batch[:, :, 3] - bboxes_batch[:, :, 1], min=0, max=1)
        target_cx = (bboxes_batch[:, :, 0] + bboxes_batch[:, :, 2]).view(bboxes_size) / 2
        target_cy = (bboxes_batch[:, :, 1] + bboxes_batch[:, :, 3]).view(bboxes_size) / 2
        # print('bw: ', bw)
        # print('bh: ', bh)
        if rand_shift_gate:
            target_cx = self.random_shift(target_cx, bw / 2)
            target_cy = self.random_shift(target_cy, bh / 2)
        # print('target_cx: ', target_cx[0])
        # print('target_cy: ', target_cy[0])
        # target_y = target_y - 0.05
        tx = (0.5 - target_cx) * 2
        ty = (0.5 - target_cy) * 2
        # print("tx, ty: ", tx, ty)

        # -----------------------Scale--------------------------
        scale = torch.sqrt(bw * bh * self.scale_rate).view(bboxes_size) # [0, 1]
        scale = scale * patch_ratio
        # print('scale shape: ', scale)

        # ----------------Random Rotate-------------------------
        angle = torch.zeros(bboxes_size).to(self.device)
        if rand_rotate_gate:
            angle = angle.uniform_(self.min_rotate_angle, self.max_rotate_angle)
        # print('angle shape:', angle.shape)
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Ready for the affine matrix
        theta = torch.FloatTensor(bboxes_size, 2, 3).fill_(0)
        # print(cos, scale)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        s = adv_patch_batch.size()
        adv_patch_batch = adv_patch_batch.view(bboxes_size, s[2], s[3], s[4])
        # print('adv batch view', adv_patch_batch.shape)
        grid = F.affine_grid(theta.cuda(), adv_patch_batch.shape)
        adv_patch_batch_t = F.grid_sample(adv_patch_batch, grid)

        return adv_patch_batch_t.view(batch_size, lab_len, s[2], s[3], s[4])


class PatchRandomApplier(nn.Module):
    def __init__(self, device, rotate_angle=20, rand_loc_rate=0.1, scale_rate=0.2):
        """

        :param rotate_angle: random rotate angle range from [-rotate_angle, rotate_angle]
        :param rand_loc_rate: random shift rate (of patch) range
        :param scale_rate: patch size / bbox size
        """
        super().__init__()
        self.patch_transformer = PatchTransformer(device, rotate_angle, rand_loc_rate, scale_rate)
        self.patch_applier = PatchApplier()
        self.device = device

    def list2tensor(self, list_batch, max_len=10):
        """This made this class an agent, the outside funcs don't have to care about the processing
        of the bbox format, and the PatchTransformer only need to process the uniformed bbox torch tensor batch.

        :param bboxes:
        :return:
        """
        bboxes_tensor_batch = None
        for i, bboxes_list in enumerate(list_batch):
            # print(f'batch {i}', len(bboxes_list))
            if type(bboxes_list) is np.ndarray or type(bboxes_list) is list:
                bboxes_list = torch.FloatTensor(bboxes_list)
            print(bboxes_list.size(0))
            if bboxes_list.size(0) == 0:
                padded_lab = torch.zeros((max_len, 6)).unsqueeze(0).to(self.device)
            else:
                bboxes_list = bboxes_list[:max_len + 1]
                pad_size = max_len - len(bboxes_list)
                # print(bboxes_list, pad_size)
                padded_lab = F.pad(bboxes_list, (0, 0, 0, pad_size), value=0).unsqueeze(0)

            if bboxes_tensor_batch is None:
                bboxes_tensor_batch = padded_lab
            else:
                bboxes_tensor_batch = torch.cat((bboxes_tensor_batch, padded_lab))
        # print('list2tensor :', bboxes_tensor_batch.shape)
        return bboxes_tensor_batch

    def forward(self, img_batch, adv_patch, bboxes_batch, jitter_gate=True, rotate_gate=True, shift_gate=False):
        """ This func to process the bboxes list of mini-batch into uniform torch.tensor and
        apply the patch into the img batch. Every patch stickers will be randomly transformed
        by given transform range before being attached.

        :param img_batch:
        :param adv_patch:
        :param bboxes_batch: bbox [batch_size, [N*6]]
        :return:
        """

        # print(img_batch.size, adv_patch.size)
        batch_size = img_batch.size(0)
        pad_size = (img_batch.size(-1) - adv_patch.size(-1)) / 2
        padding = nn.ConstantPad2d((int(pad_size + 0.5), int(pad_size), int(pad_size + 0.5), int(pad_size)), 0)  # (LRTB)

        if isinstance(bboxes_batch, list):
            bboxes_batch = self.list2tensor(bboxes_batch)
        lab_len = bboxes_batch.size(1)
        # --------------Median pool blur & Random jitter---------------------
        adv_patch = self.patch_transformer.median_pooler(adv_patch).unsqueeze(0)  # [1, 1, 3, N, N]
        adv_patch_batch = adv_patch.expand(batch_size, lab_len, -1, -1, -1)
        if jitter_gate:
            adv_patch_batch = self.patch_transformer.random_jitter(adv_patch_batch)
        adv_patch_batch = padding(adv_patch_batch)

        # print('adv batch padded: ', adv_batch.shape)
        patch_ratio = img_batch.size(-1)/adv_patch.size(-1)
        adv_batch_t = self.patch_transformer(adv_patch_batch, bboxes_batch, patch_ratio,
                                             rand_rotate_gate=rotate_gate, rand_shift_gate=shift_gate)

        adv_img_batch = self.patch_applier(img_batch, adv_batch_t)
        # print('Patch apply out: ', adv_img_batch.shape)
        return adv_img_batch


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            # print(adv.shape)
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch
