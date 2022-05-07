import argparse

import os

from attackAPI import *

from tools.utils import *


if __name__ == '__main__':
    save_root = './results/military'
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack_method', type=str, default='parallel')
    parser.add_argument('--config_file', type=str, default='parallel20.yaml')
    # parser.add_argument('--postfix', type=str, default='')
    args = parser.parse_args()
    cfg = yaml.load(open('./configs/'+args.config_file), Loader=yaml.FullLoader)
    cfg = obj(cfg)
    data_root = './data/military_data/JPEGImages'
    device = torch.device("cuda")
    detector_attacker = UniversalDetectorAttacker(cfg, device)
    img_names = [os.path.join(data_root, i) for i in os.listdir(data_root)]
    save_plot = True

    save_patch_name = args.config_file.split('.')[0] + '.png'
    # save_patch_name = None
    attack(cfg, img_names, detector_attacker, save_patch_name, save_root, save_plot, args.attack_method)