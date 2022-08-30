import kornia
import torch
import torch.nn.functional as F
import math
from .convertor import FormatConverter
from .adv import MedianPool2d


class DataTransformer(torch.nn.Module):
    def __init__(self, device, rand_rotate=20, rand_zoom_in=0.3, rand_brightness=0.2, rand_saturation=0.3):
        super().__init__()
        self.device = device
        self.rand_rotate_angle = rand_rotate
        self.rand_rotate = rand_rotate / 180 * math.pi
        self.rand_zoom_in = rand_zoom_in
        self.rand_brightness = rand_brightness
        self.rand_saturation = rand_saturation
        self.median_pooler = MedianPool2d(7, same=True)

    def jitter(self, patch, min_contrast=0.8, max_contrast=1.2,
                  min_brightness=-0.1, max_brightness=0.1, noise_factor = 0.10):
        adv_patch = self.median_pooler(patch)
        # Create random contrast tensor
        contrast = torch.FloatTensor(1).uniform_(min_contrast, max_contrast).to(self.device)
        contrast = contrast.expand(1, adv_patch.size(-3), adv_patch.size(-2), adv_patch.size(-1))

        # Create random brightness tensor
        brightness = torch.FloatTensor(1).uniform_(min_brightness, max_brightness).to(self.device)
        brightness = brightness.expand(1, adv_patch.size(-3), adv_patch.size(-2), adv_patch.size(-1))

        # Create random noise tensor
        noise = torch.FloatTensor(adv_patch.size()).uniform_(-1, 1) * noise_factor
        noise = noise.to(self.device)

        # Apply contrast/brightness/noise, clamp
        adv_patch = contrast * adv_patch + brightness + noise
        adv_patch = torch.clamp(adv_patch, 0., 1.)
        # adv_patch = kornia.augmentation.RandomRotation(self.rand_rotate_angle)(adv_patch)
        # adv_patch = self.rand_affine_matrix(adv_patch)
        return adv_patch

    def rand_affine_matrix(self, img_tensor):
        tx = 0; ty = 0
        batch_size = img_tensor.size(0)
        # rotate
        angle = torch.zeros(batch_size)
        angle = angle.uniform_(-self.rand_rotate, self.rand_rotate)
        # print('angle shape:', angle.shape)
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # scale
        # scale = torch.FloatTensor(batch_size).uniform_(1-self.rand_zoom_in, 1+self.rand_zoom_in)
        scale = 1

        theta = torch.FloatTensor(batch_size, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        grid = F.affine_grid(theta.cuda(), img_tensor.shape)
        img_tensor_t = F.grid_sample(img_tensor, grid)

        return img_tensor_t

    def forward(self, img_tensor, jiter=True, rotate=False):
        batch_size = img_tensor.size(0)
        img_tensor_t = img_tensor

        if jiter:
            gate = torch.zeros(1).bernoulli_(0.5).item()
            gate = 0
            if gate == 0:
                img_tensor_t = kornia.augmentation.RandomGaussianNoise(mean=0., std=.01, p=1)(img_tensor_t)
                factor = torch.FloatTensor(batch_size).fill_(0).uniform_(0, self.rand_brightness)
                img_tensor_t = kornia.enhance.adjust_brightness(img_tensor_t, factor, clip_output=True)
                # img_tensor_t = kornia.enhance.adjust_contrast(img_tensor_t, factor, clip_output=True)

            gate = torch.zeros(1).bernoulli_(0.5).item()
            gate = 0
            if gate == 0:
                # img_tensor_t = kornia.filters.motion_blur(img_tensor_t, 5, torch.tensor([90., 180, ]),
                #                                           torch.tensor([1., -1.]))
                factor = torch.FloatTensor(batch_size).fill_(0).uniform_(1 - self.rand_saturation, 1 + self.rand_saturation)
                img_tensor_t = kornia.enhance.adjust_saturation(img_tensor_t, factor)

            gate = torch.zeros(1).bernoulli_(0.5).item()
            gate = 0
            # if gate == 0:
                # crop_trans = kornia.augmentation.RandomResizedCrop((416, 416), (0.8, 0.8), (1.2, 1.2))
                # img_tensor_t = crop_trans(img_tensor_t)
            # img_tensor_t = self.rand_affine_matrix(img_tensor_t)

        if rotate:
            img_tensor_t = kornia.augmentation.RandomRotation(self.rand_rotate_angle)(img_tensor_t)

        torch.clamp_(img_tensor_t, min=0, max=1)
        return img_tensor_t