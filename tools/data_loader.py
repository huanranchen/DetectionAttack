import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class detDataSet(Dataset):
    def __init__(self, images_path, input_size):
        self.imgs = [os.path.join(images_path, i) for i in os.listdir(images_path)]
        self.input_size = input_size
        self.n_samples = len(self.imgs)

    def __getitem__(self, index):
        """ Reading image """
        bgr_img_numpy = cv2.imread(self.imgs[index])
        bgr_img_numpy = cv2.resize(bgr_img_numpy, self.input_size)
        image = cv2.cvtColor(bgr_img_numpy, cv2.COLOR_BGR2RGB)

        # auto scale(/255) when dtype=uint8
        image = transforms.ToTensor()(image.astype(np.uint8))

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


from torchvision import transforms
import cv2, os
import torch
from PIL import Image
from operator import itemgetter
import torch.nn.functional as F
import fnmatch
from torch.utils.data import Dataset


class InriaDataset(Dataset):
    def __init__(self, img_dir, lab_dir, img_size, data_dealer, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_size = img_size
        self.img_paths = []
        self.data_dealer = data_dealer
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        sizes = [Image.open(f, 'r').size for f in self.img_paths]
        self.max_im_width = max(sizes, key=itemgetter(0))[0]
        self.max_im_height = max(sizes, key=itemgetter(1))[1]
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = 0
        for lab_file in self.lab_paths:
            with open(lab_file) as f:
                for i, l in enumerate(f):
                    pass
            line_count = i + 1
            self.max_n_labels = max(line_count, self.max_n_labels)
        # print(self.max_n_labels)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        ori_image = Image.open(img_path).convert('RGB')
        label = torch.from_numpy(np.loadtxt(lab_path))
        self.tmp = lab_path
        self.tmp_label = label
        if label.dim() == 1:
            label = label.unsqueeze(0)
        # print(img_path, lab_path, ori_image.size, label.shape)
        # image, label = self.pad_and_scale(ori_image, label)

        image, _ = self.data_dealer(img_path=img_path)
        # transform = transforms.ToTensor()
        # image = transform(image)

        # TODO: lab is not handled
        # label = self.pad_lab(label)
        return image#, label  # , np.array(ori_image)

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w, h = img.size
        if w == h:
            padded_img = img
        else:
            dim_to_pad = 1 if w < h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                if lab.shape[-1] > 0:
                    lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                    lab[:, [3]] = (lab[:, [3]] * w / h)
                # print('lab: ', self.tmp_label, self.tmp, lab.shape)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                if lab.shape[-1] > 0:
                    lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                    lab[:, [4]] = (lab[:, [4]] * h / w)
        resize = transforms.Resize(self.img_size)
        padded_img = resize(padded_img)  # choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        print('max: ', self.max_n_labels)
        if (pad_size > 0):
            print(lab)
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=1)
            print(padded_lab)
        else:
            padded_lab = lab
        return padded_lab


def init_data_loader(cfg):
    data_loader = torch.utils.data.DataLoader(
        InriaDataset(cfg.DETECTOR.IMG_DIR, cfg.DETECTOR.LAB_DIR,
                     cfg.DETECTOR.INPUT_SIZE, shuffle=True),
        batch_size=cfg.DETECTOR.BATCH_SIZE,
        shuffle=True,
        num_workers=10)
    return data_loader

