import sys
import numpy as np
import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from natsort import natsorted

class DetDatasetLab(Dataset):
    def __init__(self, images_path, lab_path, input_size):
        self.im_path = images_path
        self.lab_path = lab_path
        self.imgs = natsorted(filter(lambda p: p.endswith('.png'), os.listdir(images_path)))
        self.input_size = input_size
        self.n_samples = len(self.imgs)
        self.max_n_labels = 10
        self.ToTensor = transforms.Compose([
            transforms.Resize(self.input_size),
            transforms.ToTensor()
        ])

    def pad_im(self, img, lab):
        w, h = img.size
        if w == h:
            return img

        pad_size = int((w - h) / 2)
        if pad_size < 0:
            pad = (abs(pad_size), 0)
            side_len = h
            lab[:, [1, 3]] = (lab[:, [1, 3]] * w + pad_size) / h
        else:
            side_len = w
            lab[:, [2, 4]] = (lab[:, [2, 4]] * h + pad_size) / w
            pad = (0, pad_size)

        padded_img = Image.new('RGB', (side_len, side_len), color=(127, 127, 127))
        padded_img.paste(img, pad)
        print('loader pad & scale: ', lab)
        return padded_img, lab

    def pad_lab(self, lab):
        lab = torch.cat(
            (lab[:, 1:], torch.ones(len(lab)).unsqueeze(1), torch.zeros(len(lab)).unsqueeze(1)),
            1
        )
        # print('loader pad lab: ', lab)
        pad_size = self.max_n_labels - lab.shape[0]
        if (pad_size > 0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=0)
            # padded_lab[-pad_size:, -1] = -1
        else:
            padded_lab = lab
        return padded_lab

    def __getitem__(self, index):
        lab_path = os.path.join(self.lab_path, self.imgs[index].replace('png', 'txt'))
        im_path = os.path.join(self.im_path, self.imgs[index])

        lab = np.loadtxt(lab_path) if os.path.getsize(lab_path) else np.zeros(5)
        lab = torch.from_numpy(lab).float()
        if lab.dim() == 1:
            lab = lab.unsqueeze(0)
        lab = lab[:self.max_n_labels]

        image = Image.open(im_path).convert('RGB')
        # image, lab = self.pad_im(image, lab)

        return self.ToTensor(image), self.pad_lab(lab)

    def __len__(self):
        return self.n_samples


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
                transforms.RandomChoice(subpolicy1)
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
        image = self.pad_scale(image)

        return self.ToTensor(image)

    def __len__(self):
        return self.n_samples


def check_valid(name):
    return name.endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))



def read_img_np_batch(names, input_size):
    # read (RGB unit8) numpy img batch from names list and rescale into input_size
    # return: numpy, uint8, RGB, [0, 255], NCHW
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


def dataLoader(data_root, lab_root=None, input_size=None, batch_size=1, is_augment=False,
               shuffle=False, pin_memory=False, num_workers=16, sampler=None):
    if input_size is None:
        input_size = [416, 416]
    if lab_root is None:
        data_set = DetDataset(data_root, input_size, is_augment=is_augment)
    else:
        data_set = DetDatasetLab(data_root, lab_root, input_size)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, sampler=sampler)
    return data_loader