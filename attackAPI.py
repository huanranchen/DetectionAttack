import torch
import yaml
import math
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

from detector_lab.HHDet.utils import init_detector

from attacks.bim import LinfBIMAttack
from attacks.mim import LinfMIMAttack
from attacks.pgd import LinfPGDAttack

from losses import temp_attack_loss
from tools.utils import obj, load_class_names, process_shape
from tools.det_utils import plot_boxes_cv2, inter_nms
from tools.data_handler import read_img_np_batch

import warnings

warnings.filterwarnings('ignore')

attacker_dict = {
    "bim": LinfBIMAttack,
    "mim": LinfMIMAttack,
    "pgd": LinfPGDAttack,
}


def init_detectors(detector_names):
    detectors = []

    for detector_name in detector_names:
        detector = init_detector(detector_name)

        detectors.append(detector)

    return detectors


class DetctorAttacker(object):
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.detectors = init_detectors(cfg.DETECTOR.NAME)
        self.patch_boxes = []
        self.class_names = load_class_names(self.cfg.DETECTOR.CLASS_NAME_FILE)

    def init_attaker(self):
        cfg = self.cfg
        self.attacker = attacker_dict[cfg.ATTACKER.METHOD](loss_fuction=temp_attack_loss, model=self.detectors,
                                                           norm='L_infty',
                                                           epsilons=cfg.ATTACKER.EPSILON,
                                                           max_iters=cfg.ATTACKER.MAX_ITERS,
                                                           step_size=cfg.ATTACKER.STEP_SIZE,
                                                           class_id=cfg.ATTACKER.TARGET_CLASS)

    def plot_boxes(self, img, boxes, savename=None):
        # print(boxes)
        plot_boxes_cv2(img, np.array(boxes), self.class_names, savename=savename)

    def patch_pos(self, preds, ratio=-1):
        height, width = self.cfg.DETECTOR.INPUT_SIZE
        patch_boxs = []
        # for every bbox in the img
        for pred in preds:
            # print('get pos', pred)
            x1, y1, x2, y2, conf, id = pred
            # print('bbox: ', x1, y1, x2, y2)
            if self.cfg.ATTACKER.ATTACK_CLASS == -1 or self.cfg.ATTACKER.ATTACK_CLASS == int(id):
                p_x1 = ((x1 + x2) / 2) - ((math.sqrt(self.cfg.ATTACKER.PATCH_ATTACK.SCALE) * (x2 - x1)) / 2)
                p_x1 = int(p_x1.clip(0, 1) * width)
                p_y1 = ((y1 + y2) / 2) - ((math.sqrt(self.cfg.ATTACKER.PATCH_ATTACK.SCALE) * (y2 - y1)) / 2)
                p_y1 = int(p_y1.clip(0, 1) * height)
                p_x2 = ((x1 + x2) / 2) + ((math.sqrt(self.cfg.ATTACKER.PATCH_ATTACK.SCALE) * (x2 - x1)) / 2)
                p_x2 = int(p_x2.clip(0, 1) * width)
                p_y2 = ((y1 + y2) / 2) + ((math.sqrt(self.cfg.ATTACKER.PATCH_ATTACK.SCALE) * (y2 - y1)) / 2)
                p_y2 = int(p_y2.clip(0, 1) * height)
                p_x1, p_y1, p_x2, p_y2 = process_shape(p_x1, p_y1, p_x2, p_y2, ratio=ratio)
                patch_boxs.append([p_x1, p_y1, p_x2, p_y2])
                # print('rectify bbox:', [p_x1, p_y1, p_x2, p_y2])
        return patch_boxs

    def get_patch_pos(self, preds, img_cv2):
        patch_boxs = self.patch_pos(preds)
        self.patch_boxes += patch_boxs
        # print("self.patch_boxes", patch_boxs)

    def init_patches(self):
        for i in range(len(self.patch_boxes)):
            p_x1, p_y1, p_x2, p_y2 = self.patch_boxes[i][:4]
            patch = np.random.randint(low=0, high=255, size=(p_y2 - p_y1, p_x2 - p_x1, 3))
            self.patch_boxes[i].append(patch)  # x1, y1, x2, y2, patch

    def add_patches(self, img_tensor, detector, is_normalize=True):
        for i in range(len(self.patch_boxes)):
            p_x1, p_y1, p_x2, p_y2 = self.patch_boxes[i][:4]
            if is_normalize:
                # init the patches
                patch_tensor, patch_cv2 = detector.normalize(self.patch_boxes[i][-1])
            else:
                # when attacking
                patch_tensor = self.patch_boxes[i][-1].detach()
                patch_tensor.requires_grad = True
            self.patch_boxes[i][-1] = patch_tensor  # x1, y1, x2, y2, patch_tensor
            img_tensor[0, :, p_y1:p_y2, p_x1:p_x2] = patch_tensor
        return img_tensor


class UniversalDetectorAttacker(DetctorAttacker):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.universal_patch = None

    def init_universal_patch(self, height=100, width=100):
        universal_patch = np.random.randint(low=0, high=255, size=(height, width, 3))
        universal_patch = np.expand_dims(np.transpose(universal_patch, (2, 0, 1)), 0)
        self.universal_patch = torch.from_numpy(np.array(universal_patch, dtype='float32') / 255.).to(self.device)
        # self.universal_patch.requires_grad = True

    def get_patch_pos_batch(self, all_preds):
        # get all bboxs of setted target. If none target bbox is got, return has_target=False
        # img_cv2: [b, h, w, c]
        self.batch_patch_boxes = []
        has_target = False
        # print(self.universal_patch.shape)
        h, w = self.universal_patch.shape[-2:]
        patch_ratio = w/h
        for index, preds in enumerate(all_preds):
            # for every image in the batch
            patch_boxs = self.patch_pos(preds, ratio=patch_ratio)
            self.batch_patch_boxes.append(patch_boxs)
            if len(patch_boxs) > 0:
                has_target = True
        return has_target

    def add_universal_patch(self, numpy_batch, detector, is_normalize=True, universal_patch=None):
        if universal_patch is None:
            universal_patch = self.universal_patch
        img_tensor = detector.init_img_batch(numpy_batch)

        if is_normalize:
            # init the patches
            universal_patch = detector.normalize_tensor(universal_patch)
        else:
            # when attacking
            universal_patch = universal_patch.detach_()
            universal_patch.requires_grad = True
        # print(len(img_tensor))
        for i, img in enumerate(img_tensor):
            # print('adding patch, ', i, 'got box num:', len(self.batch_patch_boxes[i]))
            for j, bbox in enumerate(self.batch_patch_boxes[i]):
                # for jth bbox in ith-img's bboxes
                p_x1, p_y1, p_x2, p_y2 = bbox[:4]
                patch_tensor = F.interpolate(universal_patch, size=(p_y2 - p_y1, p_x2 - p_x1), mode='bilinear')
                img_tensor[i][:, p_y1:p_y2, p_x1:p_x2] = patch_tensor[0]
        return img_tensor, universal_patch

    def merge_batch_pred(self, all_preds, preds):
        if all_preds is None:
            # if no box detected, that dimention will be a array([])
            all_preds = preds
            # print(preds[0].shape)
        else:
            for i, (all_pred, pred) in enumerate(zip(all_preds, preds)):
                pred = np.array(pred)
                if all_pred.shape[0] and pred.shape[0]:
                    all_preds[i] = np.r_[all_pred, pred]
                    continue
                if all_pred.shape[0]:
                    all_preds[i] = all_pred
                else:
                    all_preds[i] = pred
                # else:
                #     all_preds[i] = np.r_[all_pred, np.array([])]
                #     # print(all_preds)

        # if all_confs is None:
        #     all_confs = confs
        # else:
        #     all_confs = torch.cat((all_confs, confs), 1)
        return all_preds

    def imshow_save(self, img_numpy_batch, save_path, save_name, detectors=None):
        if detectors is None:
            detectors = self.detectors
        os.makedirs(save_path, exist_ok=True)
        for detector in detectors:
            tmp, _ = self.add_universal_patch(img_numpy_batch, detector)
            preds, _ = detector.detect_img_batch_get_bbox_conf(tmp)
            img_numpy, img_numpy_int8 = detector.unnormalize(tmp[0])
            self.plot_boxes(img_numpy_int8, preds[0],
                            savename=os.path.join(save_path, save_name))

    def save_patch(self, save_path, save_patch_name, patch=None):
        import cv2
        save_path = save_path + '/patch/'
        os.makedirs(save_path, exist_ok=True)
        save_patch_name = os.path.join(save_path, save_patch_name)
        if patch is None:
            patch = self.universal_patch.clone()
        tmp = patch.cpu().detach().squeeze(0).numpy()
        tmp *= 255.
        tmp = np.transpose(tmp, (1, 2, 0))
        tmp = cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR)
        tmp = tmp.astype(np.uint8)
        # print(tmp.shape)
        # cv2.imwrite('./patch.png', tmp)
        cv2.imwrite(save_patch_name, tmp)
        print('saving patch to ' + save_patch_name)

    def serial_attack(self, img_numpy_batch):
        for detector in self.detectors:
            patch = self.attacker.serial_non_targeted_attack(img_numpy_batch, self, detector)
            self.universal_patch = patch

    def parallel_attack(self, img_numpy_batch):
        patch_updates = torch.zeros(self.universal_patch.shape).to(self.device)
        for detector in self.detectors:
            # adv_img_tensor = self.add_universal_patch(img_numpy_batch, detector)
            # print('------------------', detector.name)
            patch_update = self.attacker.parallel_non_targeted_attack(img_numpy_batch, self, detector)
            patch_updates += patch_update
        # print('self.universal: ', self.universal_patch)
        self.universal_patch += patch_updates / len(self.detectors)


def attack(cfg, img_names, detector_attacker, save_name, save_path = './results',
           save_plot=False, attack_method='parallel'):

    # init the adversarial patches
    detector_attacker.init_universal_patch()
    detector_attacker.init_attaker()
    init_plot = False

    for epoch in range(cfg.ATTACKER.MAX_ITERS):
        for index in tqdm(range(0, len(img_names), cfg.DETECTOR.BATCH_SIZE)):
            names = img_names[index:index + cfg.DETECTOR.BATCH_SIZE]
            img_numpy_batch = read_img_np_batch(names, cfg.DETECTOR.INPUT_SIZE)
            # print(img_numpy_batch.shape)
            all_preds = None
            for detector in detector_attacker.detectors:
                img_tensor_batch = detector.init_img_batch(img_numpy_batch)
                preds, _ = detector.detect_img_batch_get_bbox_conf(img_tensor_batch)
                all_preds = detector_attacker.merge_batch_pred(all_preds, preds)

            all_preds = inter_nms(all_preds)
            # get position of adversarial patches
            has_target = detector_attacker.get_patch_pos_batch(all_preds)
            if not has_target:
                continue

            if init_plot and save_plot:
                init_plot = False
                detector_attacker.imshow_save(img_numpy_batch, save_path, 'original_'+save_name)

            if attack_method == 'parallel':
                detector_attacker.parallel_attack(img_numpy_batch)
            elif attack_method == 'serial':
                detector_attacker.serial_attack(img_numpy_batch)

            if save_plot:
                for detector in detector_attacker.detectors:
                    detector_attacker.imshow_save(img_numpy_batch, os.path.join(save_path, detector.name),
                                                  save_name, detectors=[detector])

        detector_attacker.save_patch(save_path, str(epoch)+'_'+save_name)


if __name__ == '__main__':
    import argparse

    save_root = './results/inria'
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--attack_method', type=str, default='serial')
    parser.add_argument('-cfg', '--cfg', type=str, default='inria.yaml')
    args = parser.parse_args()

    cfg = yaml.load(open('./configs/'+args.cfg), Loader=yaml.FullLoader)
    cfg = obj(cfg)
    data_root = './data'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector_attacker = UniversalDetectorAttacker(cfg, device)
    data_root = cfg.DETECTOR.IMG_DIR
    img_names = [os.path.join(data_root, i) for i in os.listdir(data_root)]
    init_plot = True
    save_plot = True

    save_patch_name = args.cfg.split('.')[0] + '.png'
    attack(cfg, img_names, detector_attacker, save_patch_name, save_root, save_plot, args.attack_method)
