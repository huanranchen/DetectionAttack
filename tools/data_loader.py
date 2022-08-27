import numpy as np
import cv2
import torch
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kornia
import math


class DataTransformer(torch.nn.Module):
    def __init__(self, device, rand_rotate=20, rand_zoom_in=0.3, rand_brightness=0.2, rand_saturation=0.3):
        super().__init__()
        self.device = device
        self.rand_rotate_angle = rand_rotate
        self.rand_rotate = rand_rotate / 180 * math.pi
        self.rand_zoom_in = rand_zoom_in
        self.rand_brightness = rand_brightness
        self.rand_saturation = rand_saturation

    def jitter(self, adv_patch, min_contrast=0.8, max_contrast=1.2,
                  min_brightness=-0.1, max_brightness=0.1, noise_factor = 0.10):
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
        # print(adv_patch.shape)
        # adv_patch = kornia.augmentation.RandomRotation(self.rand_rotate_angle)(adv_patch)
        # print(adv_patch.shape)
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
        scale = torch.FloatTensor(batch_size).uniform_(1-self.rand_zoom_in, 1+self.rand_zoom_in)

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
            gate = torch.zeros(3).bernoulli_(0.5).item()
            if gate[0] == 0:
                img_tensor_t = kornia.augmentation.RandomGaussianNoise(mean=0., std=.01, p=1)(img_tensor_t)
                factor = torch.FloatTensor(batch_size).fill_(0).uniform_(0, self.rand_brightness)
                img_tensor_t = kornia.enhance.adjust_brightness(img_tensor_t, factor, clip_output=True)
                img_tensor_t = kornia.enhance.adjust_contrast(img_tensor_t, factor, clip_output=True)

            if gate[2] == 0:
                img_tensor_t = kornia.filters.motion_blur(img_tensor_t, 5, torch.tensor([90., 180, ]),
                                                          torch.tensor([1., -1.]))
                factor = torch.FloatTensor(batch_size).fill_(0).uniform_(1 - self.rand_saturation, 1 + self.rand_saturation)
                img_tensor_t = kornia.enhance.adjust_saturation(img_tensor_t, factor)

            if gate[1] == 0:
                crop_trans = kornia.augmentation.RandomResizedCrop((416, 416), (0.08, 1.0), (0.75, 1.33))
                img_tensor_t = crop_trans(img_tensor_t)
            # img_tensor_t = self.rand_affine_matrix(img_tensor_t)

        if rotate:
            img_tensor_t = kornia.augmentation.RandomRotation(self.rand_rotate_angle)(img_tensor_t)

        torch.clamp_(img_tensor_t, min=0, max=1)
        return img_tensor_t


class DetDataset(Dataset):
    def __init__(self, images_path, input_size, is_augment=False):
        self.imgs = [os.path.join(images_path, i) for i in os.listdir(images_path)]
        self.input_size = input_size
        self.n_samples = len(self.imgs)
        # is_augment = False
        self.transform = transforms.Compose([])
        if is_augment:
            subpolicy1 = [
                transforms.CenterCrop(256),
                transforms.RandomResizedCrop(416, scale=(0.8, 1)),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.3),
            ]
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomEqualize(p=0.3),
                transforms.RandomChoice(subpolicy1),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
            ])
        self.ToTensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])

    def pad_scale(self, img):
        w, h = img.size
        if w == h:
            return img

        pad_size = int((w - h) / 2)
        if pad_size < 0:
            pad = (abs(pad_size), 0)
            side_len = h
        else:
            side_len = w
            pad = (0, pad_size)

        padded_img = Image.new('RGB', (side_len, side_len), color=(127, 127, 127))
        padded_img.paste(img, pad)
        return padded_img

    def __getitem__(self, index):
        # print(self.imgs[index], index)
        image = Image.open(self.imgs[index]).convert('RGB')
        image = self.transform(image)
        image = self.ToTensor(self.pad_scale(image))

        return image

    def __len__(self):
        return self.n_samples


def check_valid(name):
    return name.endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))



def read_img_np_batch(names, input_size):
    # read (RGB unit8) numpy img batch from names list and rescale into input_size
    # return: numpy, uint8, RGB, [0, 255], NCHW
    # TODO: assert (names)
    img_numpy_batch = None
    for name in names:
        if not check_valid(name):
            print(f'{name} is invalid image format!')
            continue
        bgr_img_numpy = cv2.imread(name)
        # print('img: ', input_size)
        bgr_img_numpy = cv2.resize(bgr_img_numpy, input_size)
        img_numpy = cv2.cvtColor(bgr_img_numpy, cv2.COLOR_BGR2RGB)

        img_numpy = np.expand_dims(np.transpose(img_numpy, (2, 0, 1)), 0)

        # print(img_numpy.shape)
        if img_numpy_batch is None:
            img_numpy_batch = img_numpy
        else:
            img_numpy_batch = np.concatenate((img_numpy_batch, img_numpy), axis=0)
    return img_numpy_batch


def dataLoader(data_root, input_size=None, batch_size=1, is_augment=False,
               shuffle=False, pin_memory=False, num_workers=16, sampler=None):
    if input_size is None:
        input_size = [416, 416]
    data_set = DetDataset(data_root, input_size, is_augment=is_augment)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, sampler=sampler)
    return data_loader
