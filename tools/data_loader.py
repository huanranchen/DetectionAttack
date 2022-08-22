import numpy as np
import cv2
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class DataTransformer(torch.nn.Module):
    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size
        self.subpolicy1 = [
            transforms.CenterCrop(256),
            transforms.RandomResizedCrop(416, scale=(0.8, 0.9)),
        ]
        self.subpolicy2 = [
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomEqualize(p=0.3),
        ]
        # self.transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomChoice(subpolicy1),
        #     transforms.RandomChoice(subpolicy2)
        # ])

    def forward(self, img_tensor):
        for batch_i, im in enumerate(img_tensor):
            # print(torch.max(im))
            im = torch.tensor(im * 255., dtype=torch.uint8)
            trans = transforms.Compose([
                     transforms.RandomHorizontalFlip(p=0.5),
                     transforms.RandomChoice(self.subpolicy1),
                     transforms.RandomChoice(self.subpolicy2),
                     transforms.Resize(self.image_size),
                    ])
            img_tensor[batch_i] = torch.tensor(trans(im) / 255., dtype=torch.float16)
        return img_tensor


class detDataSet(Dataset):
    def __init__(self, images_path, input_size, is_augment=False):
        self.imgs = [os.path.join(images_path, i) for i in os.listdir(images_path)]
        self.input_size = input_size
        self.n_samples = len(self.imgs)
        is_augment = False
        if is_augment:
            subpolicy1 = [
                transforms.CenterCrop(256),
                transforms.RandomResizedCrop(416, scale=(0.8, 0.9)),
            ]
            subpolicy2 = [
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            ]
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomEqualize(p=0.3),
                transforms.RandomChoice(subpolicy1),
                transforms.RandomChoice(subpolicy2),
                transforms.Resize(self.input_size),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor()
            ])

    def __getitem__(self, index):
        """ Reading image """
        # print(self.imgs[index], index)
        image = Image.open(self.imgs[index]).convert('RGB')
        image = self.transform(image)
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
    data_set = detDataSet(data_root, input_size, is_augment=is_augment)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=pin_memory, sampler=sampler)
    return data_loader
