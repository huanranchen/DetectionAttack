import os
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
import torch
import numpy as np

import sys, os
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from detector_lab import init_detectors
# from evaluate import UniversalPatchEvaluator
from tools import ConfigParser
from tools.data_loader import detDataSet
from tools.parser import load_class_names


class Utils:
    def __init__(self, cfg):
        self.class_names = load_class_names(os.path.join(PROJECT_DIR, cfg.DETECTOR.CLASS_NAME_FILE))

    def save_label(self, preds, save_path, save_name, save_conf=True):
        save_name = save_name.split('.')[0] + '.txt'
        save_to = os.path.join(save_path, save_name)
        s = []
        for pred in preds:
            # N*6: x1, y1, x2, y2, conf, cls
            x1, y1, x2, y2, conf, cls = pred

            try:
                cls = self.class_names[int(cls)].replace(' ', '')
            except Exception as e:
                print(pred, int(cls))
                assert 1 == 0
            tmp = [cls, float(x1), float(y1), float(x2), float(y2)]
            if save_conf:
                tmp.insert(1, float(conf))
                # print('tmp', tmp)

            tmp = [str(i) for i in tmp]
            s.append(' '.join(tmp))

        with open(save_to, 'w') as f:
            f.write('\n'.join(s))


if __name__ == "__main__":
    source = 'Train'
    parser = argparse.ArgumentParser()
    parser.add_argument('-dr', '--data_root', type=str, default=f"{PROJECT_DIR}/data/INRIAPerson/{source}/pos")
    parser.add_argument('-sr', '--save_root', type=str, default=f'{PROJECT_DIR}/data/INRIAPerson/{source}/labels/')
    parser.add_argument('-cfg', '--config_file', type=str, default=f'eva.yaml')
    args = parser.parse_args()

    args.config_file = os.path.join(f'{PROJECT_DIR}/configs', args.config_file)
    cfg = ConfigParser(args.config_file)
    detectors = init_detectors(cfg.DETECTOR.NAME, cfg)
    utils = Utils(cfg)
    device = torch.device('cuda')
    # evaluator = UniversalPatchEvaluator(cfg, args, device)

    batch_size = 1
    print('dataroot:', os.getcwd(), args.data_root)
    img_names = [os.path.join(args.data_root, i) for i in os.listdir(args.data_root)]

    data_set = detDataSet(args.data_root, cfg.DETECTOR.INPUT_SIZE)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)

    save_path = args.save_root
    for detector in detectors:
        fp = os.path.join(save_path, detector.name + '-labels')
        os.makedirs(fp, exist_ok=True)
        for index, img_tensor_batch in tqdm(enumerate(data_loader)):
            names = img_names[index:index + batch_size]
            img_name = names[0].split('/')[-1]
            all_preds = None

            img_tensor_batch = img_tensor_batch.to(detector.device)
            preds, _ = detector.detect_img_batch_get_bbox_conf(img_tensor_batch)

            # print(fp)
            utils.save_label(preds[0], fp, img_name, save_conf=False)