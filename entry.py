import argparse
import torch
import yaml
import os

from tools.utils import obj
from attackAPI import attack, UniversalDetectorAttacker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--attack_method', type=str, default='serial')
    parser.add_argument('-cfg', '--cfg', type=str, default='inria.yaml')
    parser.add_argument('-s', '--save_root', type=str, default='./results/inria')
    parser.add_argument('-p', '--plot_save', action='store_true')
    parser.add_argument('--cuda', type=str, default='0')
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

    cfg = yaml.load(open('./configs/'+args.cfg), Loader=yaml.FullLoader)
    cfg = obj(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    detector_attacker = UniversalDetectorAttacker(cfg, device)
    data_root = cfg.DETECTOR.IMG_DIR
    img_names = [os.path.join(data_root, i) for i in os.listdir(data_root)]

    prefix = args.cfg.split('.')[0]
    save_patch_name = prefix + '.png'
    patch_name = attack(cfg, img_names, detector_attacker, save_patch_name, args.save_root,
           args.plot_save, args.attack_method)

    torch.cuda.empty_cache()
    cmd = 'CUDA_VISIBLE_DEVICES={} python evaluate.py \
        -p ./results/inria/patch/{} \
        -cfg ./configs/{}.yaml \
        -gt /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/labels \
        -dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/pos \
        -s /home/chenziyan/work/BaseDetectionAttack/data/inria'.format(args.cuda, patch_name, prefix)
    os.system(cmd)

if __name__ == '__main__':
    main()