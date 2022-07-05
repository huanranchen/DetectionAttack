import argparse
import os
import torch
from tqdm import tqdm

from tools.parser import ConfigParser
from tools.data_loader import read_img_np_batch
from losses import temp_attack_loss
from evaluate import UniversalPatchEvaluator



def get_loss(args, patch, total_step = 1, use_which_image = None):
    batch_size = 32 if use_which_image is None else 1
    # read config file
    cfg = ConfigParser(args.config_file)
    device = torch.device('cuda')
    evaluator = UniversalPatchEvaluator(cfg, args, device, if_read_patch=False)
    evaluator.read_patch_from_memory(patch)

    # read imgs
    img_names = [os.path.join(args.data_root, i) for i in os.listdir(args.data_root)]
    img_names.sort()
    if use_which_image is not None:
        img_names = [img_names[use_which_image]]
    step = 0
    total_loss = 0
    for index in tqdm(range(0, len(img_names), batch_size)):
        names = img_names[index:index + batch_size]
        img_name = names[0].split('/')[-1]
        img_numpy_batch = read_img_np_batch(names, cfg.DETECTOR.INPUT_SIZE)

        all_preds = None
        for detector in evaluator.detectors:
            # print(detector.name)
            all_preds = None
            img_tensor_batch = detector.init_img_batch(img_numpy_batch)

            # 在干净样本上得到所有的目标检测框，定位patch覆盖的位置
            preds, detections_with_grad = detector.detect_img_batch_get_bbox_conf(img_tensor_batch)
            all_preds = evaluator.merge_batch_pred(all_preds, preds)

            # 可以拿到loss的时机1：在干净样本上的loss
            loss1 = temp_attack_loss(detections_with_grad)
            # print(loss1, detections_with_grad.shape)

            has_target = evaluator.get_patch_pos_batch(all_preds)
            if not has_target:
                continue

            # 添加patch，生成对抗样本
            adv_tensor_batch, patch_tmp = evaluator.add_universal_patch(img_numpy_batch, detector)
            # 对对抗样本进行目标检测
            preds, detections_with_grad = detector.detect_img_batch_get_bbox_conf(adv_tensor_batch)

            loss2 = temp_attack_loss(detections_with_grad)
            # print(loss2, detections_with_grad.shape)
            total_loss += loss2.item()
            step += 1
            if step > total_step:
                return total_loss / step

    return total_loss / step


if __name__ == '__main__':
    from Draws.Landscape import train_valid_3dlandscape

    train_valid_3dlandscape()
