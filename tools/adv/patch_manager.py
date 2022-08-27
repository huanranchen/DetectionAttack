import numpy as np
import torch
import cv2
from PIL import Image

from ..convertor import FormatConverter


class PatchManager:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.patch = None

    def init(self, patch_file=None):
        init_mode = self.cfg.INIT
        if patch_file is None:
            self.generate(init_mode)
        else:
            self.read(patch_file)
        self.patch = self.ori_patch

    def total_variation(self):
        pass

    def update_(self, patch_new):
        del self.patch
        self.patch = patch_new

    def clamp_(self, p_min=0, p_max=1):
        torch.clamp_(self.patch.data, min=p_min, max=p_max)

    def read(self, patch_file):
        print('Reading patch from file: ' + patch_file)
        if patch_file.endswith('.pth'):
            patch = torch.load(patch_file, map_location=self.device)
            # patch.new_tensor(patch)
            print(patch.shape, patch.requires_grad, patch.is_leaf)
        else:
            patch = Image.open(patch_file).convert('RGB')
            patch = FormatConverter.PIL2tensor(patch)
        if patch.ndim == 3:
            patch = patch.unsqueeze(0)
        self.ori_patch = patch.to(self.device)

    def generate(self, init_mode='random'):
        height = self.cfg.HEIGHT
        width = self.cfg.WIDTH
        if init_mode.lower() == 'random':
            print('Random initializing a universal patch')
            patch = torch.rand((1, 3, height, width))
        elif init_mode.lower() == 'gray':
            print('Gray initializing a universal patch')
            patch = torch.full((1, 3, height, width), 0.5)
        self.ori_patch = patch.to(self.device)

    def patch_clone(self):
        del self.patch
        self.patch = self.ori_patch.clone()