import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import math
import numpy as np

from .median_pool import MedianPool2d


class PatchTransformer(nn.Module):
    def __init__(self, device, cfg_patch, rotate_angle=20, rand_shift_rate=0.4):
        """ This will be used while applying patch to the bbox both in Training and Testing.

        :param cfg_patch: cfg.ATTACKER.PATCH from the cfg object
        :param rotate_angle: random rotate angle range from [-rotate_angle, rotate_angle]
        :param rand_loc_rate: random shift rate (of patch) range
        :param scale_rate: patch size / bbox size
        """
        super().__init__()
        self.min_rotate_angle = -rotate_angle / 180 * math.pi
        self.max_rotate_angle = rotate_angle / 180 * math.pi
        self.rand_shift_rate = rand_shift_rate
        self.scale_rate = cfg_patch.SCALE
        self.cfg_patch = cfg_patch
        self.median_pooler = MedianPool2d(7, same=True)
        self.device = device
        self.logger = True

    def forward(self, adv_patch_batch, bboxes_batch, patch_ori_size, rand_rotate_gate=True, rand_shift_gate=False, p9_scale=True, rdrop=False):
        """
        apply patches.
        : param bboxes_batch: batchsize, num_bboxes_in_each_image, size6([x1, y1, x2, y2, conf, cls_id])
        """

        batch_size = bboxes_batch.size(0)
        lab_len = bboxes_batch.size(1)
        bboxes_size = np.prod([batch_size, lab_len]) # np.product. just a number
        # print(bboxes_batch[0][:4, :])

        # ------------------Rand drop----------------------
        if rdrop:
            drop_gate = torch.cuda.FloatTensor(torch.Size((batch_size, lab_len))).uniform_(0, 3.5).byte()
            drop_gate[drop_gate>1] = 1
            # print(drop_gate, drop_gate.size())
            drop_gate = drop_gate.unsqueeze(-1).expand(-1, -1, 6)
            # print(drop_gate)
            bboxes_batch *= drop_gate

        # -------------Shift & Random relocate--------------
        # bbox format is [x1, y1, x2, y2, conf, cls_id]
        bw = bboxes_batch[:, :, 2] - bboxes_batch[:, :, 0]
        bh = bboxes_batch[:, :, 3] - bboxes_batch[:, :, 1]
        target_cx = (bboxes_batch[:, :, 0] + bboxes_batch[:, :, 2]).view(bboxes_size) / 2
        target_cy = (bboxes_batch[:, :, 1] + bboxes_batch[:, :, 3]).view(bboxes_size) / 2

        if rand_shift_gate:
            target_cx = self.random_shift(target_cx, bw / 2)
            target_cy = self.random_shift(target_cy, bh / 2)
        if p9_scale:
            # print(target_cy.size(), bh.size())
            target_cy -= bh.view(bboxes_size) * 0.1
        tx = (0.5 - target_cx) * 2
        ty = (0.5 - target_cy) * 2
        # print("tx, ty: ", tx, ty)

        # -----------------------Scale--------------------------
        bw *= adv_patch_batch.size(-1)
        bh *= adv_patch_batch.size(-2)
        # p9_scale = False
        if p9_scale:
            target_size = self.scale_rate * torch.sqrt((bw ** 2) + (bh ** 2)).view(bboxes_size)
        else:
            target_size = torch.sqrt(bw * bh * self.scale_rate).view(bboxes_size)  # [0, 1]

        scale = target_size / patch_ori_size
        if hasattr(self.cfg_patch, 'scale_max') and self.cfg_patch.scale_max > 0:
            scale = torch.clamp(scale, max=self.cfg_patch.scale_max, min=0)
        # print('scale shape: ', scale)

        # ----------------Random Rotate-------------------------
        angle = torch.cuda.FloatTensor(bboxes_size).fill_(0)
        if rand_rotate_gate:
            angle = angle.uniform_(self.min_rotate_angle, self.max_rotate_angle)
        # print('angle shape:', angle.shape)
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # ----------Ready for the affine matrix-------------
        theta = torch.cuda.FloatTensor(bboxes_size, 2, 3).fill_(0)
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
        grid = F.affine_grid(theta, adv_patch_batch.shape)
        adv_patch_batch_t = F.grid_sample(adv_patch_batch, grid)

        return adv_patch_batch_t.view(s[0], s[1], s[2], s[3], s[4])

    def random_shift(self, x, limited_range):
        shift = limited_range * torch.cuda.FloatTensor(x.size()).uniform_(-self.rand_shift_rate, self.rand_shift_rate)
        return x + shift

    def cutout(self, x, cutout_ratio=0.4, cutout_fill=0.5, rand_shift=-0.05, level='instance', p_erase=0.9):
        """
        Random erase(or Cut out) area of the adversarial patches.
        :param x: adversarial patches in a mini-batch.
        :param cutout_fill(>0): cutout area to fill with what magnitude.(0 is the backround)
        :param erase_size:
        :return:
        """
        if self.logger:
            self.logger = False
            print('Cutout level: ', level, '; cutout ratio: ', cutout_ratio, '; random shift: ', rand_shift)

        gate = torch.tensor([0]).bernoulli_(p_erase)
        if gate.item() == 0: return x
        assert cutout_fill > 0, 'Error! The cutout area can\'t be filled with 0'
        s = x.size()
        batch_size = s[0]
        lab_len = s[1]
        bboxes_shape = torch.Size((batch_size, lab_len))
        bboxes_size = np.prod([batch_size, lab_len])

        if level == "instance":
            target_size = bboxes_size
        elif level == "image":
            target_size = batch_size
        elif level == 'batch':
            target_size = 1
        # print(bboxes_size, target_size)

        bg = torch.cuda.FloatTensor(bboxes_shape).fill_(cutout_fill)
        bg = self.equal_size(bg, x.size)

        angle = torch.cuda.FloatTensor(target_size).fill_(0)
        if level != 'instance':
            angle = angle.unsqueeze(-1).expand(s[0], s[1]).reshape(-1)
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        target_cx = torch.cuda.FloatTensor(target_size).uniform_(rand_shift, 1-rand_shift)
        target_cy = torch.cuda.FloatTensor(target_size).uniform_(rand_shift, 1-rand_shift)
        if level != 'instance':
            target_cx = target_cx.unsqueeze(-1).expand(s[0], s[1]).reshape(-1)
            target_cy = target_cy.unsqueeze(-1).expand(s[0], s[1]).reshape(-1)
        tx = (0.5 - target_cx) * 2
        ty = (0.5 - target_cy) * 2

        # TODO: This assumes the patch is in a square-shape
        scale = cutout_ratio
        theta = torch.cuda.FloatTensor(bboxes_size, 2, 3).fill_(0)
        # print(cos, scale)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        bg = bg.view(bboxes_size, s[2], s[3], s[4])
        x = x.view(bboxes_size, s[2], s[3], s[4])
        # print('adv batch view', adv_patch_batch.shape)
        grid = F.affine_grid(theta, bg.shape)
        bg = F.grid_sample(bg, grid)

        # bg_mag = cutout_fill-1
        # bg_fill = torch.ones_like(bg) * bg_mag
        # cutout_fill = torch.ones_like(bg) * cutout_fill
        # bg = torch.where(bg == 0, bg_fill, bg)
        # bg = torch.where(bg == 1, cutout_fill, bg)

        # print(bg.size(), x.size())
        x_t = torch.where((bg == 0), x, bg)
        return x_t.view(s[0], s[1], s[2], s[3], s[4])

    def equal_size(self, tensor, size):
        tensor = tensor.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        tensor = tensor.expand(-1, -1, size(-3), size(-2), size(-1))
        return tensor

    def random_shuffle(self, x):
        batchsize, channels, height, width = x.size()
        groups = 1
        channels_per_group = int(channels / groups)
        x_s = x.view(batchsize, groups, channels_per_group, height, width)
        x_s = x_s.transpose(1, 2).contiguous()
        x_s = x_s.view(batchsize, -1, height, width)
        return x_s

    def random_jitter(self, x, min_contrast=0.8, max_contrast=1.2, min_brightness=-0.1, max_brightness=0.1, noise_factor = 0.10):
        bboxes_shape = torch.Size((x.size(0), x.size(1)))
        contrast = torch.cuda.FloatTensor(bboxes_shape).uniform_(min_contrast, max_contrast)
        contrast = self.equal_size(contrast, x.size)

        # Create random brightness tensor
        brightness = torch.cuda.FloatTensor(bboxes_shape).uniform_(min_brightness, max_brightness)
        brightness = self.equal_size(brightness, x.size)

        # Create random noise tensor
        noise = torch.cuda.FloatTensor(x.size()).uniform_(-1, 1) * noise_factor

        # Apply contrast/brightness/noise, clamp
        x = contrast * x + brightness + noise
        # x = torch.clamp(x, 0.000001, 0.99999)
        return x