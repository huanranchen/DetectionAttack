import numpy as np
import torch
import cv2
from PIL.Image import Image

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
        # if self.patch.is_leaf:
        #     self.patch.requires_grad = True

    def update(self, patch_new):
        del self.patch
        self.patch = patch_new

    def read(self, patch_file):
        print('Reading patch from file: ' + patch_file)
        if patch_file.endswith('.pth'):
            patch = torch.load(patch_file, map_location=self.device)
            if patch.ndim == 3:
                patch = patch.unsqueeze(0)
            # patch.new_tensor(patch)
            print(patch.shape, patch.requires_grad, patch.is_leaf)
        else:
            patch = Image.open(patch_file).convert('RGB')
            patch = FormatConverter.PIL2tensor(patch)

        self.patch = patch.to(self.device)

    def generate(self, init_mode='random'):
        height = self.cfg.HEIGHT
        width = self.cfg.WIDTH
        if init_mode.lower() == 'random':
            print('Random initializing a universal patch')
            patch = np.random.randint(low=0, high=255, size=(3, height, width))
        elif init_mode.lower() == 'gray':
            print('Gray initializing a universal patch')
            patch = np.ones((3, height, width)) * 127.5

        self.patch = FormatConverter.numpy2tensor(patch).to(self.device)