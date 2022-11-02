import copy
import torch
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[0]


def parser_input():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', '--cfg', type=str)
    parser.add_argument('-lp', '--label_path', type=str, default="data/INRIAPerson/Test/labels/")
    parser.add_argument('-dr', '--data_root', type=str, default="data/INRIAPerson/Test/pos/")
    parser.add_argument('-p', '--patch_dir', type=str)
    parser.add_argument('-s', '--save', type=str)
    parser.add_argument('-ng', '--gen_no_label', action='store_true')
    args = parser.parse_args()

    args.patch_dir = os.path.join(PROJECT_DIR, args.patch_dir)
    args.label_path = os.path.join(PROJECT_DIR, args.label_path)
    args.data_root = os.path.join(PROJECT_DIR, args.data_root)
    # args.save = os.path.join(args.save, args.cfg)

    args.cfg = './configs/' + args.cfg

    cfg = ConfigParser(args.cfg)
    args.eva_class = cfg.ATTACKER.ATTACK_CLASS
    # args.label_path = os.path.join(ROOT, cfg.DATA.TRAIN.LAB_DIR)
    args.save = os.path.join(PROJECT_DIR, args.save)
    os.makedirs(args.save, exist_ok=True)
    # args.data_root = os.path.join(ROOT, cfg.DATA.TRAIN.IMG_DIR)
    args.test_origin = False
    args.detectors = None
    args.stimulate_uint8_loss = False
    args.save_imgs = False
    args.quiet = True

    return args, cfg


def readAP(p):
    with open(p, 'r') as f:
        mAP = f.readlines()[1].split('%')[0]
        # print(mAP)
    return float(mAP)


if __name__ == '__main__':
    from tools.parser import ConfigParser
    import os
    import numpy as np
    import re

    import matplotlib.pyplot as plt
    from evaluate import eval_patch, get_save

    args, cfg = parser_input()
    print('save dir: ', args.save)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    patch_files = os.listdir(args.patch_dir)
    rule = '.png'
    patch_files = list(filter(lambda file: rule in file, patch_files))
    print('patch num: ', len(patch_files))

    y = {}
    for detector_name in cfg.DETECTOR.NAME:
        y[detector_name.lower()] = []

    x = []
    for patch_file in patch_files:
        args_tmp = copy.deepcopy(args)
        prefix = patch_file.split('.')[0]
        x.append(int(prefix.split('_')[-1]))
        args_tmp.patch = os.path.join(args_tmp.patch_dir, *['all_data', patch_file])
        args_tmp = get_save(args_tmp)
        det_mAPs, _, _, _ = eval_patch(args_tmp, cfg)

        for k, v in det_mAPs.items():
            y[k].append(float(v))

    np.save(args.save+'/x.npy', x)
    np.save(args.save + '/y.npy', y)

    plt.figure()
    for ny in y.values():
        # print(x, train_y)
        # plt.plot(x, train_y)
        # print(x, train_y, test_y)
        plt.scatter(x, ny, c='r', label='Test')
    plt.legend()
    plt.ylabel('AP(%)')
    plt.xlabel('# iteration')
    plt.savefig(args.save+'/serial_ap.png', dpi=300)
    print(args.save+'/gap.png')