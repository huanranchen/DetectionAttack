import torch
import os
import time
import numpy as np
from tqdm import tqdm

from tools import save_tensor
from tools.draw import VisualBoard
from tools.loader import dataLoader
from train_pgd import logger


def attack(cfg, data_root, detector_attacker, save_name, args=None):
    def get_iter():
        return (epoch - 1) * len(data_loader) + index * cfg.DETECTOR.BATCH_SIZE

    logger(cfg, args)
    data_sampler = None
    detector_attacker.init_universal_patch(args.patch)
    data_loader = dataLoader(data_root, lab_root=cfg.DATA.TRAIN.LAB_DIR,
                             input_size=cfg.DETECTOR.INPUT_SIZE, is_augment=False,
                             batch_size=cfg.DETECTOR.BATCH_SIZE, sampler=data_sampler, shuffle=True)
    detector_attacker.gates = {'jitter': True, 'median_pool': True, 'rotate': True,
                               'shift': False, 'p9_scale': True}
    p_obj = detector_attacker.patch_obj.patch
    p_obj.requires_grad = True
    optimizer = torch.optim.Adam([p_obj], lr=cfg.ATTACKER.START_LEARNING_RATE, amsgrad=True)

    from torch import optim
    scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=100)
    scheduler = scheduler_factory(optimizer)
    detector_attacker.attacker.set_optimizer(optimizer)

    start_index = int(args.patch.split('/')[-1].split('_')[0]) if args.patch is not None else 1
    loss_array = []
    save_tensor(detector_attacker.universal_patch, f'{save_name}', args.save_path)

    vlogger = None
    if not args.debugging:
        vlogger = VisualBoard(optimizer, name=args.board_name, start_iter=start_index)
        detector_attacker.vlogger = vlogger
    for epoch in range(start_index, cfg.ATTACKER.MAX_ITERS+1):
        et0 = time.time()
        ep_loss = 0
        for index, (img_tensor_batch, lab) in enumerate(tqdm(data_loader, desc=f'Epoch {epoch}')):
            if vlogger: vlogger(epoch, get_iter())
            img_tensor_batch = img_tensor_batch.to(detector_attacker.device)
            detector_attacker.all_preds = lab.to(detector_attacker.device)
            # print(detector_attacker.all_preds)

            loss = detector_attacker.attack(img_tensor_batch, mode='optim')
            # print('                 loss : ', loss)
            ep_loss += loss

        if epoch % 10 == 0:
            # patch_name = f'{epoch}_{save_name}'
            patch_name = f'{save_name}'
            save_tensor(detector_attacker.universal_patch, patch_name, args.save_path)
            print('Saving patch to ', os.path.join(args.save_path, patch_name))

        et1 = time.time()
        ep_loss /= len(data_loader)
        scheduler.step(ep_loss)
        if vlogger:
            vlogger.write_ep_loss(ep_loss)
            vlogger.write_scalar(et1-et0, 'misc/ep time')
        # print('           ep loss : ', ep_loss)
        loss_array.append(float(ep_loss))
    np.save(os.path.join(args.save_path, 'loss.npy'), loss_array)


if __name__ == '__main__':
    from tools.parser import ConfigParser
    from BaseDetectionAttack.attack.attacker import UniversalDetectorAttacker
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, help='fine-tune from a pre-trained patch', default=None)
    parser.add_argument('-m', '--attack_method', type=str, default='optim')
    parser.add_argument('-cfg', '--cfg', type=str, default='optim.yaml')
    parser.add_argument('-n', '--board_name', type=str, default=None)
    parser.add_argument('-d', '--debugging', action='store_true')
    parser.add_argument('-s', '--save_path', type=str, default='./results/exp2/optim')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_prefix = args.cfg.split('.')[0] if args.board_name is None else args.board_name
    save_patch_name = save_prefix + '.png'
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