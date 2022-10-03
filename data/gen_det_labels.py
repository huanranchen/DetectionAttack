import os
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np

import sys, os
PWD = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PWD)
sys.path.append(PROJECT_DIR)

from detlib import init_detectors
from tools import ConfigParser
from tools.loader import dataLoader
from tools.parser import load_class_names
from tools.det_utils import plot_boxes_cv2

class Utils:
    def __init__(self, cfg):
        self.cfg = cfg
        self.class_names = load_class_names(os.path.join(PROJECT_DIR, cfg.DATA.CLASS_NAME_FILE))

    def save_label(self, preds, save_path, save_name, save_conf=True, rescale=True):
        ori_size = self.cfg.DETECTOR.INPUT_SIZE[0]
        save_name = save_name.split('.')[0] + '.txt'
        save_to = os.path.join(save_path, save_name)
        s = []
        for pred in preds:
            # N*6: x1, y1, x2, y2, conf, cls
            if rescale:
                pred[:4] *= ori_size
            x1, y1, x2, y2, conf, cls = pred
            cls = self.class_names[int(cls)].replace(' ', '')
            tmp = [cls, float(x1), float(y1), float(x2), float(y2)]
            if save_conf:
                tmp.insert(1, float(conf))
            tmp = [str(i) for i in tmp]
            s.append(' '.join(tmp))
        with open(save_to, 'w') as f:
            f.write('\n'.join(s))


if __name__ == "__main__":

    source = 'Train'
    parser = argparse.ArgumentParser()
    parser.add_argument('-dr', '--data_root', type=str, default=f"{PROJECT_DIR}/data/INRIAPerson/{source}/pos")
    parser.add_argument('-sr', '--save_root', type=str, default=f'{PROJECT_DIR}/data/INRIAPerson/{source}/labels/')
    parser.add_argument('-cfg', '--config_file', type=str, default=f'test.yaml')
    parser.add_argument('-d', '--detector_name', nargs='+', default=None)
    parser.add_argument('-nr', '--nrescale', action='store_true')
    parser.add_argument('-i', '--imgs', action='store_true')
    # parser.add_argument('-c', '--class', nargs='+', default=-1)
    args = parser.parse_args()

    args.data_root = os.path.join(PROJECT_DIR, args.data_root)
    args.save_root = os.path.join(PROJECT_DIR, args.save_root)
    args.config_file = os.path.join(f'{PROJECT_DIR}/configs', args.config_file)
    cfg = ConfigParser(args.config_file)
    if args.detector_name is not None:
        cfg.DETECTOR.NAME = args.detector_name
    detectors = init_detectors(cfg.DETECTOR.NAME, cfg)

    utils = Utils(cfg)
    device = torch.device('cuda')
    # evaluator = UniversalPatchEvaluator(cfg, args, device)

    batch_size = 1
    print('dataroot     :', os.getcwd(), args.data_root)
    img_names = [os.path.join(args.data_root, i) for i in os.listdir(args.data_root)]

    data_loader = dataLoader(data_root=args.data_root, input_size=cfg.DETECTOR.INPUT_SIZE,
                             batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    postfix = '-labels' if args.nrescale else '-rescale-labels'
    print('rescale      :', args.nrescale, postfix)
    save_path = args.save_root
    for detector in detectors:
        fp = os.path.join(save_path, detector.name + postfix)
        os.makedirs(fp, exist_ok=True)
        for index, img_tensor_batch in tqdm(enumerate(data_loader)):
            names = img_names[index:index + batch_size]
            img_name = names[0].split('/')[-1]
            all_preds = None

            img_tensor_batch = img_tensor_batch.to(detector.device)
            preds, _ = detector(img_tensor_batch)

            if args.imgs:
                save_dir = f'./test/{detector.name}'
                os.makedirs(save_dir, exist_ok=True)
                img_numpy, img_numpy_int8 = detector.unnormalize(img_tensor_batch[0])
                plot_boxes_cv2(img_numpy_int8, np.array(preds[0]), cfg.all_class_names,
                               savename=os.path.join(save_dir, img_name))
            # os.path.join('./test/' + img_name)
            # print(fp)
            utils.save_label(preds[0], fp, img_name, save_conf=False, rescale=not args.nrescale)

            exit()