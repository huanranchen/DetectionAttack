import torch
import os
import time
import numpy as np
from tqdm import tqdm

from tools import save_tensor
from tools.plot import VisualBoard
from tools.loader import dataLoader
from tools.parser import logger
from scripts.dict import scheduler_factory, optim_factory

# for validation in training
from scripts.obj_args import eva_args
from evaluate import eval_patch, get_save


def init(detector_attacker, cfg, data_root, args=None, log=True):
    if log: logger(cfg, args)

    data_sampler = None
    data_loader = dataLoader(data_root,
                             input_size=cfg.DETECTOR.INPUT_SIZE, is_augment=cfg.DATA.AUGMENT,
                             batch_size=cfg.DETECTOR.BATCH_SIZE, sampler=data_sampler, shuffle=True)

    detector_attacker.init_universal_patch(args.patch)
    detector_attacker.init_attaker()

    vlogger = None
    if log and args and not args.debugging:
        vlogger = VisualBoard(name=args.board_name, new_process=args.new_process,
                              optimizer=detector_attacker.attacker)
        detector_attacker.vlogger = vlogger

    return data_loader, vlogger


def train_uap(cfg, detector_attacker, save_name, args=None, data_root=None, save_process=True):
    def get_iter(): return (epoch - 1) * len(data_loader) + index

    if data_root is None: data_root = cfg.DATA.TRAIN.IMG_DIR
    data_loader, vlogger = init(detector_attacker, cfg, args=args, data_root=data_root)
    optimizer = optim_factory[cfg.ATTACKER.METHOD](detector_attacker.universal_patch, cfg.ATTACKER.STEP_LR)
    detector_attacker.attacker.set_optimizer(optimizer)
    scheduler = scheduler_factory[cfg.ATTACKER.LR_SCHEDULER](optimizer)

    loss_array = []
    save_tensor(detector_attacker.universal_patch, f'{save_name}' + '.png', args.save_path)
    for epoch in range(1, cfg.ATTACKER.MAX_EPOCH + 1):
        ep_loss = 0
        for index, img_tensor_batch in enumerate(tqdm(data_loader, desc=f'Epoch {epoch}')):
            # for index, (img_tensor_batch, img_tensor_batch2) in enumerate(tqdm(zip(data_loader, data_loader2), desc=f'Epoch {epoch}')):
            if vlogger: vlogger(epoch, get_iter())
            img_tensor_batch = img_tensor_batch.to(detector_attacker.device)

            all_preds = detector_attacker.detect_bbox(img_tensor_batch)
            # get position of adversarial patches
            target_nums = detector_attacker.get_patch_pos_batch(all_preds)
            if sum(target_nums) == 0: continue

            loss = detector_attacker.attack(img_tensor_batch, mode='optim')
            ep_loss += loss

        # print(ep_loss)
        ep_loss /= len(data_loader)
        scheduler.step(ep_loss=ep_loss, epoch=epoch)

        if vlogger: vlogger.write_ep_loss(ep_loss)
        loss_array.append(float(ep_loss))
        if epoch % 10 == 0:
            # patch_name = f'{epoch}_{save_name}'
            patch_name = f'{save_name}' + '.png'
            save_path = args.save_path
            if save_process:
                save_path = save_path + '/patch'
                os.makedirs(save_path, exist_ok=True)
                patch_name = f'{save_name}_{epoch}' + '.png'
            save_tensor(detector_attacker.universal_patch, patch_name, save_path)
            print('Saving patch to ', save_path)


    np.save(os.path.join(args.save_path, save_name + '-loss.npy'), loss_array)


if __name__ == '__main__':
    from tools.parser import ConfigParser
    from attack.attacker import UniversalAttacker
    import argparse
    import warnings

    warnings.filterwarnings('ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, help='fine-tune from a pre-trained patch', default=None)
    parser.add_argument('-m', '--attack_method', type=str, default='optim')
    parser.add_argument('-cfg', '--cfg', type=str, default='optim.yaml')
    parser.add_argument('-n', '--board_name', type=str, default=None)
    parser.add_argument('-d', '--debugging', action='store_true')
    parser.add_argument('-dis', '--model_distribute', action='store_true')
    parser.add_argument('-s', '--save_path', type=str, default='./results/exp2/optim')
    parser.add_argument('-np', '--new_process', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cfg = './configs/' + args.cfg
    save_patch_name = args.cfg.split('/')[-1].split('.')[0] if args.board_name is None else args.board_name

    cfg = ConfigParser(args.cfg)
    detector_attacker = UniversalAttacker(cfg, device, model_distribute=args.model_distribute)
    cfg.show_class_label(cfg.attack_list)
    train_uap(cfg, detector_attacker, save_patch_name, args)
