import argparse
import os
import cv2
import numpy as np
import torch
import shutil
from tqdm import tqdm

from attackAPI import UniversalDetectorAttacker
from data.gen_det_labels import Utils
from tools.parser import ConfigParser

# from tools.det_utils import plot_boxes_cv2

paths = {'attack-img': 'imgs',
         'det-lab': 'det-labels',
         'attack-lab': 'attack-labels',
         'det-res': 'det-res'}
GT = 'ground-truth'


def dir_check(save_path, rebuild=False):
    # if the target path exists, it will be deleted (for empty dirt) and rebuild-up
    def check(path, rebuild):
        if rebuild and os.path.exists(path):
            shutil.rmtree(path)
        try:
            os.makedirs(path, exist_ok=True)
        except:
            pass
        # print('mkdir: ', path)

    check(save_path, rebuild=rebuild)
    for detector_name in cfg.DETECTOR.NAME:
        detector_name = detector_name.lower()
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
        if patch_file.endswith('.pth'):
            universal_patch = torch.load(patch_file, map_location='cuda').unsqueeze(0)
            # universal_patch.new_tensor(universal_patch)
            print(universal_patch.shape, universal_patch.requires_grad, universal_patch.is_leaf)
        else:
            universal_patch = cv2.imread(patch_file)
            universal_patch = cv2.cvtColor(universal_patch, cv2.COLOR_BGR2RGB)
            universal_patch = np.expand_dims(np.transpose(universal_patch, (2, 0, 1)), 0)
            universal_patch = torch.from_numpy(np.array(universal_patch, dtype='float32') / 255.)
        self.universal_patch = universal_patch.to(self.device)

    # def save_label(self, preds, save_path, save_name, save_conf=True):
    #     save_name = save_name.split('.')[0] + '.txt'
    #     save_to = os.path.join(save_path, save_name)
    #     s = []
    #     for pred in preds:
    #         # N*6: x1 y1 x2 y2 conf cls
    #         x1, y1, x2, y2, conf, cls = pred
    #         cls = self.class_names[int(cls)].replace(' ', '')
    #         tmp = [cls, x1, y1, x2, y2]
    #         if save_conf:
    #             tmp.insert(1, conf)
    #             # print('tmp', tmp)
    #
    #         tmp = [str(i) for i in tmp]
    #         s.append(' '.join(tmp))
    #
    #     with open(save_to, 'w') as f:
    #         f.write('\n'.join(s))


def handle_input():
    def get_prefix(path):
        if os.sep in path:
            path = path.split(os.sep)[-1]
        return path.split('.')[0]

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str)
    parser.add_argument('-cfg', '--config_file', type=str)
    parser.add_argument('-d', '--detectors', nargs='+', default=None)
    parser.add_argument('-s', '--save', type=str, default='/home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train')
    parser.add_argument('-lp', '--label_path', help='ground truth & detector predicted labels dir', type=str, default='/home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/labels')
    parser.add_argument('-dr', '--data_root', type=str, default='/home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/pos')
    parser.add_argument('-o', '--test_origin', action='store_true')
    parser.add_argument('-ul', '--stimulate_uint8_loss', action='store_true')
    parser.add_argument('-i', '--save_imgs', help='to save attacked imgs', action='store_true')
    parser.add_argument('-g', '--gen_labels', action='store_true')
    parser.add_argument('-e', '--eva_class', type=str, default='-1') # '-1': all classes; '-2': attack seen classes(ATTACK_CLASS in cfg file); '-3': attack unseen classes(all_class - ATTACK_CLASS); or given a list by '['x:y']'/'[0]'
    parser.add_argument('-q', '--quiet', help='output none if set true', action='store_true')
    args = parser.parse_args()

    prefix = get_prefix(args.patch)
    args.save = os.path.join(args.save, prefix)

    print("save root: ", args.save)
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
    # print('Eva(Attack) classes from evaluation: ', cfg.show_class_index(args.eva_class))
    # print('Eva classes names from evaluation: ', args.eva_class)

    args.ignore_class = list(set(cfg.all_class_names).difference(set(args.eva_class)))
    if len(args.ignore_class) == 0:
        args.ignore_class = None


    print(args.detectors)
    if args.detectors is not None:
        cfg.DETECTOR.NAME = args.detectors

    return args, cfg


def generate_labels(evaluator, cfg, args, save_label=False):
    from tools.data_loader import detDataSet
    from torch.utils.data import DataLoader

    dir_check(args.save, rebuild=False)
    utils = Utils(cfg)
    batch_size = 1
    print('dataroot:', os.getcwd(), args.data_root)
    img_names = [os.path.join(args.data_root, i) for i in os.listdir(args.data_root)]

    data_set = detDataSet(args.data_root, cfg.DETECTOR.INPUT_SIZE)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True)
    save_path = args.save
    for index, img_tensor_batch in tqdm(enumerate(data_loader)):
    # for index in tqdm(range(0, len(img_names), batch_size)):
        names = img_names[index:index + batch_size]
        img_name = names[0].split('/')[-1]
    #     img_numpy_batch = read_img_np_batch(names, cfg.DETECTOR.INPUT_SIZE)

        for detector in evaluator.detectors:
            tmp_path = os.path.join(save_path, detector.name)
            # print('----------------', detector.name)
            all_preds = None
            img_tensor_batch = img_tensor_batch.to(detector.device)
            # img_tensor_batch = detector.init_img_batch(img_numpy_batch)
            # get object bbox: preds bbox list; detections_with_grad: confs tensor
            preds, _ = detector.detect_img_batch_get_bbox_conf(img_tensor_batch)

            all_preds = evaluator.merge_batch_pred(all_preds, preds)

            if save_label:
                # for saving the original detection info
                fp = os.path.join(tmp_path, paths['det-lab'])
                utils.save_label(preds[0], fp, img_name, save_conf=False)

            if args.test_origin:
                fp = os.path.join(tmp_path, paths['det-res'])
                utils.save_label(preds[0], fp, img_name, save_conf=True)

            evaluator.get_patch_pos_batch(all_preds)

            if args.save_imgs:
                # for saving the attacked imgs
                ipath = os.path.join(tmp_path, 'imgs')
                evaluator.imshow_save(img_tensor_batch, ipath, img_name, detectors=[detector])

            adv_img_tensor, _ = evaluator.apply_universal_patch(img_tensor_batch, detector)
            if args.stimulate_uint8_loss:
                adv_img_tensor = detector.int8_precision_loss(adv_img_tensor)
            preds, _ = detector.detect_img_batch_get_bbox_conf(adv_img_tensor)

            # for saving the attacked detection info
            lpath = os.path.join(tmp_path, paths['attack-lab'])
            # print(lpath)
            utils.save_label(preds[0], lpath, img_name)
            # break
        # break


if __name__ == '__main__':
    from tools.eva.main import compute_mAP
    from tools.parser import dict2txt

    args, cfg = handle_input()

    device = torch.device('cuda')
    evaluator = UniversalPatchEvaluator(cfg, args, device)
    # set the eva classes to be the ones to attack
    evaluator.attack_list = cfg.show_class_index(args.eva_class)

    if args.gen_labels:
        generate_labels(evaluator, cfg, args)
    save_path = args.save

    det_mAPs = {}
    gt_mAPs = {}
    ori_mAPs = {}
    # to compute mAP
    for detector in evaluator.detectors:
        # tmp_path = os.path.join(cfg.ATTACK_SAVE_PATH, detector.name)
        path = os.path.join(save_path, detector.name)

        # link the path of the GT labels
        gt_path = os.path.join(path, GT)
        if os.path.exists(gt_path):
            os.remove(gt_path)
            print("Exists det ground truth path, removing...")
            # os.system("sleep 1")

        cmd = 'ln -s ' + os.path.join(args.label_path, GT) + ' ' + gt_path
        # print(cmd)
        os.system(cmd)

        # link the path of the detection labels
        det_path = os.path.join(*[path, paths['det-lab']])

        if os.path.exists(det_path):
            try:
                os.rmdir(det_path)
            except:
                os.remove(det_path)
            print("Exists det label path, removing...")

        cmd = 'ln -s ' + os.path.join(args.label_path, detector.name+'-labels') + ' ' + det_path
        print(cmd)
        os.system(cmd)

        # (det-results)take clear detection results as GT label: attack results as detections
        det_mAP = compute_mAP(path=path, ignore=args.ignore_class, lab_path=paths['attack-lab'],
                                gt_path=paths['det-lab'], res_prefix='det')
        det_mAPs[detector.name] = "%.2f" % (det_mAP*100) + "%"

        # (gt-results)take original labels as GT label(default): attack results as detections
        gt_mAP = compute_mAP(path=path, ignore=args.ignore_class, lab_path=paths['attack-lab'],
                                gt_path=GT, res_prefix='gt')
        gt_mAPs[detector.name] = "%.2f" % (gt_mAP*100) + "%"

        if args.test_origin:
            rp = 'ori'
            # (ori-results)take original labels as GT label(default): clear detection res as detections
            ori_mAP = compute_mAP(path=path, ignore=args.ignore_class, lab_path=paths['det-res'],
                                gt_path=GT, res_prefix=rp)
            ori_mAPs[rp][detector.name] = "%.2f" % (ori_mAP*100) + "%"

        # merge_plot(ori_aps_dic, path, det_aps_dic, gt_aps_dic)
    print("det mAP      :", det_mAPs)
    det_mAP_file = os.path.join(save_path, 'det-mAP.txt')
    if not os.path.exists(det_mAP_file):
        with open(det_mAP_file, 'w') as f:
            f.write('scale: '+str(cfg.ATTACKER.PATCH_ATTACK.SCALE)+'\n')
            f.write('--------------------------\n')
    dict2txt(det_mAPs, det_mAP_file)
    dict2txt(gt_mAPs, os.path.join(save_path, 'gt-mAP.txt'))
