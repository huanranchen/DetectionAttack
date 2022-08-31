import argparse
import os
import sys

import cv2
import numpy as np
import torch
import shutil
from tqdm import tqdm

from attackAPI import UniversalDetectorAttacker
from data.gen_det_labels import Utils
from tools.parser import ConfigParser
from detector_lab.utils import inter_nms
from tools.eva.main import compute_mAP
from tools import save_tensor

import warnings
warnings.filterwarnings('ignore')

# from tools.det_utils import plot_boxes_cv2
label_postfix = '-rescale-labels'
paths = {'attack-img': 'imgs',
         'det-lab': 'det-labels',
         'attack-lab': 'attack-labels',
         'det-res': 'det-res',
         'ground-truth': 'ground-truth'}


def path_remove(path):
    if os.path.exists(path):
        try:
            shutil.rmtree(path) # a dir
        except:
            os.remove(path) # a symbolic link

def dir_check(save_path, child_paths, rebuild=False):
    # if the target path exists, it will be deleted (for empty dirt) and rebuild-up
    def check(path, rebuild):
        if rebuild:
            path_remove(path)
        os.makedirs(path, exist_ok=True)
    check(save_path, rebuild=rebuild)
    for child_path in child_paths:
        child_path = child_path.lower()
        tmp_path = os.path.join(save_path, child_path)
        for path in paths.values():
            ipath = os.path.join(tmp_path, path)
            check(ipath, rebuild)


class UniversalPatchEvaluator(UniversalDetectorAttacker):
    def __init__(self, cfg, patch_path=None,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                 ):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.device = device
        if patch_path is not None:
            self.patch_obj.read(patch_path)

    def read_patch_from_memory(self, patch):
        self.patch_obj.update_(patch)


def handle_input():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default=None)
    parser.add_argument('-cfg', '--cfg', type=str, default=None)
    parser.add_argument('-d', '--detectors', nargs='+', default=None)
    parser.add_argument('-s', '--save', type=str, default='/home/chenziyan/work/BaseDetectionAttack/data/inria/')
    parser.add_argument('-lp', '--label_path', help='ground truth & detector predicted labels dir', type=str, default='/home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/labels')
    parser.add_argument('-dr', '--data_root', type=str, default='/home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/pos')
    parser.add_argument('-to', '--test_origin', action='store_true')
    parser.add_argument('-tg', '--test_gt', action='store_true')
    parser.add_argument('-ul', '--stimulate_uint8_loss', action='store_true')
    parser.add_argument('-i', '--save_imgs', help='to save attacked imgs', action='store_true')
    parser.add_argument('-g', '--gen_labels', action='store_true')
    parser.add_argument('-e', '--eva_class', type=str, default='-1') # '-1': all classes; '-2': attack seen classes(ATTACK_CLASS in cfg file); '-3': attack unseen classes(all_class - ATTACK_CLASS); or given a list by '['x:y']'/'[0]'
    parser.add_argument('-q', '--quiet', help='output none if set true', action='store_true')
    args = parser.parse_args()

    print("save root: ", args.save)
    cfg = ConfigParser(args.cfg)

    print(args.detectors)
    if args.detectors is not None:
        cfg.DETECTOR.NAME = args.detectors

    return args, cfg


def get_save(args):
    def get_prefix(path):
        if os.sep in path:
            path = path.split(os.sep)[-1]
        return path.split('.')[0]
    prefix = get_prefix(args.patch)
    args.save = os.path.join(args.save, prefix)
    return args


def ignore_class(args, cfg):
    # Be careful of the so-called 'attack_list' and 'eva_class' in the evaluate.py
    # For higher reusability of the codes, these variable names may be confusing
    # In this file, the 'attack_list' is loaded from the config file which has been used for training
    # (cuz we don't bother to create a new config file for evaluations)
    # Thus the 'attack_list' refers to the original attacked classes when training the patch
    # while the 'eva_list' denotes the class list to be evaluated, which are to attack in the evaluation
    # (When the eva classes are different from the original attack classes,
    # it is mainly for the partial attack in evaluating unseen-class/cross-class performance)
    args.eva_class_list = cfg.rectify_class_list(args.eva_class, dtype='str')
    # print('Eva(Attack) classes from evaluation: ', cfg.show_class_index(args.eva_class_list))
    # print('Eva classes names from evaluation: ', args.eva_class_list)

    args.ignore_class = list(set(cfg.all_class_names).difference(set(args.eva_class_list)))
    if len(args.ignore_class) == 0: args.ignore_class = None
    return args


def generate_labels(evaluator, cfg, args, save_label=False):
    from tools.loader import dataLoader

    dir_check(args.save, cfg.DETECTOR.NAME, rebuild=False)
    utils = Utils(cfg)
    batch_size = 1
    img_names = [os.path.join(args.data_root, i) for i in os.listdir(args.data_root)]

    data_loader = dataLoader(args.data_root, input_size=cfg.DETECTOR.INPUT_SIZE,
                             batch_size=batch_size, is_augment=False, pin_memory=True)
    save_path = args.save
    aspect_ratio = args.aspect_ratio if hasattr(args, 'aspect_ratio') else None
    accs_total = {}
    for detector in evaluator.detectors:
        accs_total[detector.name] = []
    # print(evaluator.detectors)
    for index, img_batch in enumerate(tqdm(data_loader, total=len(data_loader))):

        names = img_names[index:index + batch_size]
        img_name = names[0].split('/')[-1]
        # print(img_name)
        for detector in evaluator.detectors:
            # print(detector.name)
            # make sure every detector detect in a new batch of img tensors (avoid of the inplace)
            img_tensor_batch = img_batch.to(evaluator.device)
            tmp_path = os.path.join(save_path, detector.name)
            all_preds = detector(img_tensor_batch)['bbox_array']
            if save_label:
                # for saving the original detection info
                fp = os.path.join(tmp_path, paths['det-lab'])
                utils.save_label(all_preds[0], fp, img_name, save_conf=False, rescale=True)

            if hasattr(args, 'test_origin') and args.test_origin:
                fp = os.path.join(tmp_path, paths['det-res'])
                utils.save_label(all_preds[0], fp, img_name, save_conf=True, rescale=True)

            target_nums_clean = evaluator.get_patch_pos_batch(all_preds, aspect_ratio=aspect_ratio)[0]
            adv_img_tensor = evaluator.uap_apply(img_tensor_batch, eval=True)

            if hasattr(args, 'stimulate_uint8_loss') and args.stimulate_uint8_loss:
                adv_img_tensor = detector.int8_precision_loss(adv_img_tensor)

            preds = detector(adv_img_tensor)['bbox_array']
            if hasattr(args, 'save_imgs') and args.save_imgs:
                # for saving the attacked imgs
                ipath = os.path.join(tmp_path, 'imgs')
                evaluator.plot_boxes(adv_img_tensor[0], preds[0], save_path=ipath, save_name=img_name)

            # for saving the attacked detection info
            lpath = os.path.join(tmp_path, paths['attack-lab'])
            utils.save_label(preds[0], lpath, img_name, rescale=True)

            if target_nums_clean:
                target_nums_adv = 0
                if preds[0].numel():
                    target_adv = evaluator.filter_bbox(preds[0])
                    target_nums_adv = len(target_adv)
                # print('--------adv: ', target_adv, target_nums_adv)
                acc = target_nums_clean - target_nums_adv
                acc = 0 if acc < 0 else acc / target_nums_clean
                # print('acc: ', acc)
                accs_total[detector.name].append(round(acc*100, 2))
            # break
        # break
    for detector in evaluator.detectors:
        accs_total[detector.name] = np.mean(accs_total[detector.name])
    return accs_total


def init(args, cfg, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    # preprocessing the cfg
    args = get_save(args)
    args = ignore_class(args, cfg)
    evaluator = UniversalPatchEvaluator(cfg, args.patch, device)

    cfg = cfg_save_modify(cfg)
    return args, cfg, evaluator


def cfg_save_modify(cfg):
    cfg.DETECTOR.PERTURB.GATE = None
    cfg.DATA.AUGMENT = False
    return cfg


def eva(args, cfg):
    args, cfg, evaluator = init(args, cfg)
    print('------------------ Evaluating ------------------')
    print("              cfg file : ", args.cfg)
    print("             data root : ", args.data_root)
    print("            label root : ", args.label_path)
    print("             save root : ", args.save)
    # set the eva classes to be the ones to attack
    evaluator.attack_list = cfg.show_class_index(args.eva_class_list)

    accs_dict = None
    if args.gen_labels:
        accs_dict = generate_labels(evaluator, cfg, args)
    save_path = args.save

    det_mAPs = {}; gt_mAPs = {}; ori_mAPs = {}
    quiet = args.quiet if hasattr(args, 'quiet') else False
    # to compute mAP
    for detector in evaluator.detectors:
        # tmp_path = os.path.join(cfg.ATTACK_SAVE_PATH, detector.name)
        path = os.path.join(save_path, detector.name)

        # link the path of the GT labels
        gt_target = os.path.join(path, 'ground-truth')
        gt_source = os.path.join(args.label_path, paths['ground-truth']+label_postfix)
        path_remove(gt_target)
        cmd = ' '.join(['ln -s ', gt_source, gt_target])
        print(cmd)
        os.system(cmd)

        # link the path of the detection labels
        det_path = os.path.join(path, paths['det-lab'])
        path_remove(det_path)

        # cmd = 'ln -s ' + os.path.join(args.label_path, detector.name+'-labels') + ' ' + det_path
        source = os.path.join(args.label_path, detector.name + label_postfix)
        cmd = ' '.join(['ln -s ', source, det_path])
        # print(cmd)
        os.system(cmd)

        # (det-results)take clear detection results as GT label: attack results as detections
        print('ground truth     :', os.path.join(path, paths['det-lab']))
        det_mAP = compute_mAP(path=path, ignore=args.ignore_class, lab_path=paths['attack-lab'],
                                gt_path=paths['det-lab'], res_prefix='det', quiet=quiet)
        det_mAPs[detector.name] = round(det_mAP*100, 2)


        if hasattr(args, 'test_gt') and args.test_gt:
            # (gt-results)take original labels as GT label(default): attack results as detections
            # print('ground truth     :', paths['ground-truth'])
            gt_mAP = compute_mAP(path=path, ignore=args.ignore_class, lab_path=paths['attack-lab'],
                                    gt_path=paths['ground-truth'], res_prefix='gt', quiet=quiet)
            gt_mAPs[detector.name] = round(gt_mAP*100, 2)

        if hasattr(args, 'test_origin') and args.test_origin:
            rp = 'ori'
            # (ori-results)take original labels as path['ground-truth'] label(default): clear detection res as detections
            ori_mAP = compute_mAP(path=path, ignore=args.ignore_class, lab_path=paths['det-res'],
                                gt_path=paths['ground-truth'], res_prefix=rp, quiet=quiet)
            ori_mAPs[rp][detector.name] = round(ori_mAP*100, 2)

        # merge_plot(ori_aps_dic, path, det_aps_dic, gt_aps_dic)

    return det_mAPs, gt_mAPs, ori_mAPs, accs_dict


if __name__ == '__main__':
    from tools.parser import dict2txt, merge_dict_by_key

    args, cfg = handle_input()
    order = ['yolov3', 'yolov3-tiny', 'yolov4', 'yolov4-tiny', 'yolov5', 'faster_rcnn', 'ssd']
    # args, evaluator = init(args, cfg)
    det_mAPs, gt_mAPs, ori_mAPs, accs_dict = eva(args, cfg)

    det_mAP_file = os.path.join(args.save, 'det-mAP.txt')
    if not os.path.exists(det_mAP_file):
        with open(det_mAP_file, 'a') as f:
            where = cfg.ATTACKER.PATCH_ATTACK.ASPECT_RATIO
            f.write('aspect ratio   : '+str(where)+'\n')
            f.write('scale          : ' + str(cfg.ATTACKER.PATCH_ATTACK.SCALE) + '\n')
            f.write('--------------------------\n')

    det_dict = det_mAPs
    if accs_dict is not None:
        det_dict = merge_dict_by_key(det_mAPs, accs_dict)
    dict2txt(det_dict, det_mAP_file)
    dict2txt(gt_mAPs, os.path.join(args.save, 'gt-mAP.txt'))
    print("det dict      [mAP, acc] :", det_dict)
    # with open(os.path.join(args.save, 'gt-mAP.txt'), 'a'):
    #     det_mAPs['yolov3']
