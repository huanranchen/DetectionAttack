import argparse
import os
import cv2
import numpy as np
import torch
import yaml
import shutil
from tqdm import tqdm

from detector_lab.HHDet.utils import init_detector
from attackAPI import UniversalDetectorAttacker

from tools.utils import obj
from tools.mAP import draw_mAP, merge_plot
from tools.parser import ConfigParser
from tools.data_loader import read_img_np_batch
# from tools.det_utils import plot_boxes_cv2

paths = {'attack-img': 'imgs', 'det-lab': 'det-labels', 'attack-lab': 'attack-labels', 'det-res': 'det-res'}
GT = 'ground-truth'

def dir_check(save_path, rebuild=True):
    # if the target path exists, it will be deleted (for empty dirt) and rebuild-up
    def check(path, rebuild):
        if rebuild and os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)
        # print('mkdir: ', path)

    check(save_path, rebuild=rebuild)
    for detector_name in cfg.DETECTOR.NAME:
        tmp_path = os.path.join(save_path, detector_name)
        for path in paths.values():
            ipath = os.path.join(tmp_path, path)
            check(ipath, rebuild)


class UniversalPatchEvaluator(UniversalDetectorAttacker):
    def __init__(self, cfg, args, device, if_read_patch = True):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.args = args
        self.device = device
        if if_read_patch:
            self.read_patch()

    def read_patch_from_memory(self, patch):
        self.universal_patch = patch

    def read_patch(self):
        patch_file = self.args.patch
        print('reading patch ' + patch_file)
        universal_patch = cv2.imread(patch_file)
        universal_patch = cv2.cvtColor(universal_patch, cv2.COLOR_BGR2RGB)
        universal_patch = np.expand_dims(np.transpose(universal_patch, (2, 0, 1)), 0)
        self.universal_patch = torch.from_numpy(np.array(universal_patch, dtype='float32') / 255.).to(self.device)

    def save_label(self, preds, save_path, save_name, save_conf=True):
        save_name = save_name.split('.')[0] + '.txt'
        save_to = os.path.join(save_path, save_name)
        s = []
        for pred in preds:
            # N*6: x1 y1 x2 y2 conf cls
            x1, y1, x2, y2, conf, cls = pred
            cls = self.class_names[int(cls)].replace(' ', '')
            tmp = [cls, x1, y1, x2, y2]
            if save_conf:
                tmp.insert(1, conf)
                # print('tmp', tmp)

            tmp = [str(i) for i in tmp]
            s.append(' '.join(tmp))

        with open(save_to, 'w') as f:
            f.write('\n'.join(s))


def handle_input():
    def get_prefix(path):
        if os.sep in path:
            path = path.split(os.sep)[-1]
        return path.split('.')[0]

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str)
    parser.add_argument('-cfg', '--config_file', type=str)
    parser.add_argument('-d', '--detectors', type=list, default=None)
    parser.add_argument('-s', '--save', type=str, default='/home/chenziyan/work/BaseDetectionAttack/data/military_data')
    parser.add_argument('-gt', '--gt_path', help='ground truth label dir', type=str, default='/home/chenziyan/work/BaseDetectionAttack/data/military_data/AnnotationLabels')
    parser.add_argument('-dr', '--data_root', type=str, default='/home/chenziyan/work/BaseDetectionAttack/data/military_data/JPEGImages')
    parser.add_argument('-o', '--test_origin', action='store_true')
    parser.add_argument('-l', '--stimulate_uint8_loss', action='store_true')
    parser.add_argument('-i', '--save_imgs', help='to save attacked imgs', action='store_true')
    parser.add_argument('-e', '--eva_class', type=str, default='-1') # '-1': all classes; '-2': attack seen classes(ATTACK_CLASS in cfg file); '-3': attack unseen classes(all_class - ATTACK_CLASS); or given a list by '['x:y']'/'[0]'
    parser.add_argument('-q', '--quiet', help='output none if set true', action='store_true')
    args = parser.parse_args()

    prefix = get_prefix(args.patch)
    args.save = os.path.join(args.save, prefix)

    print(prefix, args.save)
    cfg = ConfigParser(args.config_file)

    # Be careful of the so-called 'attack_list' and 'eva_class' in the evaluate.py
    # For higher reusability of the codes, these variable names may be confusing
    # In this file, the 'attack_list' is loaded from the config file which has been used for training
    # (cuz we don't bother to create a new config file for evaluations)
    # Thus the 'attack_list' refers to the original attacked classes when training the patch
    # while the 'eva_list' denotes the class list to be evaluated, which are to attack in the evaluation
    # (When the eva classes are different from the original attack classes,
    # it is mainly for the partial attack in evaluating unseen-class/cross-class performance)
    args.eva_class = cfg.rectify_class_list(args.eva_class, dtype='str')
    print('Eva(Attack) classes from evaluation: ', cfg.show_class_index(args.eva_class))
    print('Eva classes names from evaluation: ', args.eva_class)

    args.ignore_class = list(set(cfg.all_class_names).difference(set(args.eva_class)))
    # print('igore: ', args.ignore_class)
    if args.detectors is not None:
        cfg.DETECTOR.NAME = args.detectors

    return args, cfg


def generate_labels(evaluator, cfg, args):
    batch_size = 1

    print('dataroot:', os.getcwd(), args.data_root)
    img_names = [os.path.join(args.data_root, i) for i in os.listdir(args.data_root)]

    save_path = args.save
    for index in tqdm(range(0, len(img_names), batch_size)):
        names = img_names[index:index + batch_size]
        img_name = names[0].split('/')[-1]
        img_numpy_batch = read_img_np_batch(names, cfg.DETECTOR.INPUT_SIZE)

        for detector in evaluator.detectors:
            tmp_path = os.path.join(save_path, detector.name)
            # print('----------------', detector.name)
            all_preds = None
            img_tensor_batch = detector.init_img_batch(img_numpy_batch)
            # get object bbox: preds bbox list; detections_with_grad: confs tensor
            preds, _ = detector.detect_img_batch_get_bbox_conf(img_tensor_batch)

            all_preds = evaluator.merge_batch_pred(all_preds, preds)
            # for saving the original detection info
            fp = os.path.join(tmp_path, paths['det-lab'])
            evaluator.save_label(preds[0], fp, img_name, save_conf=False)

            if args.test_origin:
                fp = os.path.join(tmp_path, paths['det-res'])
                evaluator.save_label(preds[0], fp, img_name, save_conf=True)

            evaluator.get_patch_pos_batch(all_preds)

            if args.save_imgs:
                # for saving the attacked imgs
                ipath = os.path.join(tmp_path, 'imgs')
                evaluator.imshow_save(img_numpy_batch, ipath, img_name, detectors=[detector])

            adv_img_tensor, _ = evaluator.add_universal_patch(img_numpy_batch, detector)
            if args.stimulate_uint8_loss:
                adv_img_tensor = detector.int8_precision_loss(adv_img_tensor)
            preds, _ = detector.detect_img_batch_get_bbox_conf(adv_img_tensor)

            # for saving the attacked detection info
            lpath = os.path.join(tmp_path, paths['attack-lab'])
            # print(lpath)
            evaluator.save_label(preds[0], lpath, img_name)
            # break
        # break


if __name__ == '__main__':
    args, cfg = handle_input()

    dir_check(args.save)
    device = torch.device('cuda')
    evaluator = UniversalPatchEvaluator(cfg, args, device)
    # set the eva classes to be the ones to attack
    evaluator.attack_list = cfg.show_class_index(args.eva_class)

    generate_labels(evaluator, cfg, args)
    save_path = args.save
    # to compute mAP
    for detector in evaluator.detectors:
        # tmp_path = os.path.join(cfg.ATTACK_SAVE_PATH, detector.name)
        path = os.path.join(save_path, detector.name)

        gt_path = os.path.join(path, GT)
        cmd = 'ln -s ' + args.gt_path + ' ' + gt_path
        os.system(cmd)

        # print(str(attack_class))
        cmd = 'python ./tools/mAP.py' + ' -p ' + path
        if len(args.ignore_class):
            cmd += ' -i ' + ' '.join(args.ignore_class)
        cmd += ' --lab-path='
        # (det-results)take clear detection results as GT label: attack results as detections
        # DET_args = {'gt_path':paths['det-lab'], 'res_prefix': 'det'}
        # DET_args.update(args)
        # det_aps_dic = draw_mAP(obj(DET_args))
        os.system(cmd + paths['attack-lab']+ ' -gt ' + paths['det-lab'] + ' -rp det')

        # (gt-results)take original labels as GT label(default): attack results as detections
        # GT_args = {'gt_path': paths['attack-lab']}
        # GT_args.update(args)
        # gt_aps_dic = draw_mAP(obj(GT_args))
        os.system(cmd + paths['attack-lab']+' -rp gt')

        if args.test_origin:
            # (ori-results)take original labels as GT label(default): clear detection res as detections
            # ORI_args = {'gt_path': paths['det-res'], 'res_prefix': 'ori'}
            # ORI_args.update(args)
            # ori_aps_dic = draw_mAP(obj(ORI_args))
            os.system(cmd + paths['det-res'] + ' -rp ori')
        # break

        # merge_plot(ori_aps_dic, path, det_aps_dic, gt_aps_dic)