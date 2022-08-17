import os
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np

from pathlib import Path
FILE = Path(__file__).resolve()
# print(__file__)
PROJECT_ROOT = str(FILE.parents[2])

import sys
sys.path.append(PROJECT_ROOT)
# print(PROJECT_ROOT, sys.path)

from tools.parser import ConfigParser
from evaluate import UniversalPatchEvaluator, eva, init, handle_input

def init_args(args, save_root, ratio):
    args.gen_labels = True
    # args.gen_labels = False
    args.eva_class = '0'
    args.test_origin = False
    args.save_imgs = True

    args.aspect_ratio = ratio
    args.save = save_root + f'/{str(ratio)[:3]}'
    print('---------ratio: ', ratio)
    return args


def draw_all_picture_of_aspect_ratio(cfg, args, postfix=''):
    aspect_ratios = np.arange(0.2, 2.1, 0.1)
    # print(aspect_ratios)
    save_root = args.save
    ys = []
    xs = []
    accs = []
    # try:
    for ratio in aspect_ratios:
        args = init_args(args, save_root, ratio)

        # if os.path.exists(args.save):
        #     print('exists, pass.')
        #     args.gen_labels = False
        # else:
        #     args.gen_labels = True

        det_mAPs, _, _, acc = eva(args, cfg)
        # print(det_mAPs.values())
        xs.append(float(str(ratio)[:3]))
        ys.append(float(list(det_mAPs.values())[0]))
        accs.append(acc)
        print(xs, ys, accs)
        np.save(f'{save_root}/{cfg.DETECTOR.NAME[0]}-xs{postfix}.npy', xs)
        np.save(f'{save_root}/{cfg.DETECTOR.NAME[0]}-ys{postfix}.npy', ys)
        np.save(f'{save_root}/{cfg.DETECTOR.NAME[0]}-accs{postfix}.npy', accs)
        # break
        plt.plot(xs, ys)
        plt.scatter(xs, ys)
        plt.ylabel('Person AP(%)')
        plt.xlabel('Aspect Ratio(%)')
        plt.savefig(f'{save_root}/{cfg.DETECTOR.NAME[0]}-mAP{postfix}.png', dpi=300)

        plt.clf()
        plt.plot(xs, accs)
        plt.scatter(xs, accs)
        plt.ylabel('Person Acc(%)')
        plt.xlabel('Aspect Ratio(%)')
        plt.savefig(f'{save_root}/{cfg.DETECTOR.NAME[0]}-acc{postfix}.png', dpi=300)

    # except Exception as e:
    #     print(e)
    #     pass



if __name__ == '__main__':
    import argparse

    targets = ['v2', 'v3', 'v3tiny', 'v4', 'v4tiny', 'v5']
    targets = ['v2']
    for target in targets:
        parser = argparse.ArgumentParser()
        parser.add_argument('-p', '--patch', type=str,
                            default=f'{PROJECT_ROOT}/results/inria/gap/aug/{target}/patch/1000_{target}-aug.png')
        parser.add_argument('-cfg', '--cfg', type=str, default=f'{target}.yaml')
        parser.add_argument('-lp', '--label_path', help='ground truth & detector predicted labels dir', type=str,
                            default=f'{PROJECT_ROOT}/data/INRIAPerson/Test/labels')
        parser.add_argument('-dr', '--data_root', type=str,
                            default=f'{PROJECT_ROOT}/data/INRIAPerson/Test/pos')
        parser.add_argument('-s', '--save', type=str,
                            default=f'{PROJECT_ROOT}/data/inria/test/settings/aspect_ratio')
        args = parser.parse_args()
        acc_fig = plt.figure()
        map_fig = plt.figure()
        args.patch = f'{PROJECT_ROOT}/results/object_score.png'
        cfg = ConfigParser(f'{PROJECT_ROOT}/configs/{args.cfg}')
        draw_all_picture_of_aspect_ratio(cfg, args, postfix='-obj_score')
