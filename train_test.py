import torch
import os
import numpy as np
from tqdm import tqdm
from detector_lab.utils import inter_nms

from tools.data_loader import dataLoader
from train import logger


def attack(cfg, data_root, detector_attacker, save_name, args=None):
    logger(cfg, args)
    save_plot = True
    data_sampler = None
    detector_attacker.init_universal_patch(args.patch)
    data_loader = dataLoader(data_root, cfg.DETECTOR.INPUT_SIZE, is_augment=cfg.DATA.AUGMENT == 1,
                             batch_size=cfg.DETECTOR.BATCH_SIZE, sampler=data_sampler)

    p_obj = detector_attacker.patch_obj.ori_patch
    p_obj.requires_grad = True
    optimizer = torch.optim.Adam([p_obj], lr=0.03, amsgrad=True)

    from torch import optim
    scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
    scheduler = scheduler_factory(optimizer)
    detector_attacker.attacker.set_optimizer(optimizer)

    start_index = int(args.patch.split('/')[-1].split('_')[0]) if args.patch is not None else 1
    loss_array = []
    detector_attacker.save_patch(args.save_path, f'{start_index-1}_{save_name}')
    for epoch in range(start_index, cfg.ATTACKER.MAX_ITERS+1):
        ep_loss = 0
        for index, img_tensor_batch in enumerate(tqdm(data_loader, desc=f'Epoch {epoch}')):
            detector_attacker.patch_obj.patch_clone()
            img_tensor_batch = img_tensor_batch.to(detector_attacker.device)

            all_preds = detector_attacker.detect_bbox(img_tensor_batch)
            # get position of adversarial patches
            target_nums = detector_attacker.get_patch_pos_batch(all_preds)
            if sum(target_nums) == 0: continue

            loss = detector_attacker.attack(img_tensor_batch, mode='optim')
            # print('                 loss : ', loss)
            ep_loss += loss
            if save_plot and index % 10 == 0:
                # for detector-specific dir name
                for detector in detector_attacker.detectors:
                    detector_attacker.adv_detect_save(img_tensor_batch,
                                                      os.path.join(args.save_path, detector.name),
                                                      save_name,
                                                      detectors=[detector])

            if index == 1 and epoch % 10 == 0:
                prefix = epoch
                patch_name = f'{prefix}_{save_name}'
                detector_attacker.save_patch(args.save_path, patch_name)

        ep_loss /= len(data_loader)
        scheduler.step(ep_loss)
        print('           ep loss : ', ep_loss)
        loss_array.append(float(ep_loss))
    np.save(os.path.join(args.save_path, 'loss.npy'), loss_array)


if __name__ == '__main__':
    from tools.parser import ConfigParser
    from attackAPI import UniversalDetectorAttacker
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, help='fine-tune from a pre-trained patch', default=None)
    parser.add_argument('-m', '--attack_method', type=str, default='optim')
    parser.add_argument('-cfg', '--cfg', type=str, default='optim.yaml')
    parser.add_argument('-s', '--save_path', type=str, default='./results/exp2/optim')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_patch_name = args.cfg.split('.')[0] + '.png'
    args.cfg = './configs/' + args.cfg

    print('-------------------------Training-------------------------')
    print('                       device : ', device)
    print('                          cfg :', args.cfg)

    cfg = ConfigParser(args.cfg)
    detector_attacker = UniversalDetectorAttacker(cfg, device)
    cfg.show_class_label(cfg.attack_list)
    data_root = cfg.DATA.TRAIN.IMG_DIR
    img_names = [os.path.join(data_root, i) for i in os.listdir(data_root)]
    attack(cfg, data_root, detector_attacker, save_patch_name, args)