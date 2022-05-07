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

from tools.utils import obj, inter_nms
from tools.data_handler import read_img_np_batch

paths = {'attack-img': 'imgs', 'det-lab': 'det-labels', 'attack-lab': 'attack-labels'}
gt = 'ground-truth'

def dir_check(cfg):
    # if the target path exists, it will be deleted (for empty dirt) and rebuild-up
    def check(path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        # print('mkdir: ', path)

    check(cfg.ATTACK_SAVE_PATH)
    for detector_name in cfg.DETECTOR.NAME:
        tmp_path = os.path.join(cfg.ATTACK_SAVE_PATH, detector_name)
        for path in paths.values():
            ipath = os.path.join(tmp_path, path)
            check(ipath)


class UniversalPatchEvaluator(UniversalDetectorAttacker):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        self.cfg = cfg
        self.device = device
        self.read_patch()

    def read_patch(self):
        print('reading patch '+self.cfg.PATCH_FILE)
        universal_patch = cv2.imread(self.cfg.PATCH_FILE)
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
            if save_conf:
                tmp = [self.class_names[int(cls)], conf, x1, y1, x2, y2]
            else:
                tmp = [self.class_names[int(cls)], x1, y1, x2, y2]
            tmp = [str(i) for i in tmp]
            s.append(' '.join(tmp))

        with open(save_to, 'w') as f:
            f.write('\n'.join(s))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--patch', type=str)
    parser.add_argument('--config_file', type=str, default='serial5.yaml')
    args = parser.parse_args()
    cfg = yaml.load(open('./configs/evaluate/'+args.config_file), Loader=yaml.FullLoader)
    cfg = obj(cfg)
    train_cfg = yaml.load(open('./configs/'+args.config_file), Loader=yaml.FullLoader)
    train_cfg = obj(train_cfg)

    device = torch.device('cuda')

    evaluator = UniversalPatchEvaluator(cfg, device)
    img_names = [os.path.join(cfg.DATA_ROOT, i) for i in os.listdir(cfg.DATA_ROOT)]
    dir_check(cfg)
    for index in tqdm(range(0, len(img_names), cfg.DETECTOR.BATCH_SIZE)):
        names = img_names[index:index + cfg.DETECTOR.BATCH_SIZE]
        img_name = names[0].split('/')[-1]
        img_numpy_batch = read_img_np_batch(names, cfg.DETECTOR.INPUT_SIZE)

        for detector in evaluator.detectors:
            tmp_path = os.path.join(cfg.ATTACK_SAVE_PATH, detector.name)
            # print('----------------', detector.name)
            all_preds = None
            img_tensor_batch = detector.init_img_batch(img_numpy_batch)
            # get object bbox: preds bbox list; detections_with_grad: confs tensor
            preds, _ = detector.detect_img_batch_get_bbox_conf(img_tensor_batch)
            all_preds = evaluator.merge_batch_pred(all_preds, preds)
            # for saving the original detection info
            fp = os.path.join(tmp_path, paths['det-lab'])
            evaluator.save_label(preds[0], fp, img_name, save_conf=False)

            evaluator.get_patch_pos_batch(all_preds)

            # for saving the attacked imgs
            ipath = os.path.join(tmp_path, 'imgs')
            evaluator.imshow_save(img_numpy_batch, ipath, img_name, detectors=[detector])

            adv_img_tensor, _ = evaluator.add_universal_patch(img_numpy_batch, detector)
            preds, _ = detector.detect_img_batch_get_bbox_conf(adv_img_tensor)

            # for saving the attacked detection info
            lpath = os.path.join(tmp_path, paths['attack-lab'])
            # print(lpath)
            evaluator.save_label(preds[0], lpath, img_name)

    # for compute mAP
    for detector in evaluator.detectors:
        # tmp_path = os.path.join(cfg.ATTACK_SAVE_PATH, detector.name)
        path = os.path.join(cfg.ATTACK_SAVE_PATH, detector.name)
        gt_path = os.path.join(path, gt)
        cmd = 'ln -s ' + cfg.GT_PATH + ' ' + gt_path
        os.system(cmd)
        cmd = 'python ./data/mAP.py -d ' + detector.name + ' -p ' + path + ' --lab_path=' + paths['attack-lab']
        # print(cmd)
        # print('test: ', os.path.join(path, paths['det-lab']))
        os.system(cmd + ' -gt ' +paths['det-lab']+ ' -rp det')
        os.system(cmd + ' -rp gt')
        # break
