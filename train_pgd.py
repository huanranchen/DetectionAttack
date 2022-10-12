import torch
import numpy as np
from tqdm import tqdm
import os
import time
from tools.plot import VisualBoard
from tools.loader import dataLoader
from tools import save_tensor
from tools.parser import logger

def modelDDP(detector_attacker, args):
    for ind, detector in enumerate(detector_attacker.detectors):
        detector_attacker.detectors[ind] = torch.nn.parallel.DistributedDataParallel(detector,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)


def attack(cfg, detector_attacker, save_name, args=None, save_step=5000):
    detector_attacker.gates = ['rotate', 'p9_scale']
    if args.random_erase: detector_attacker.gates.append('rerase')

    data_loader = dataLoader(cfg.DATA.TRAIN.IMG_DIR,
                             input_size=cfg.DETECTOR.INPUT_SIZE, is_augment='1' in cfg.DATA.AUGMENT,
                             batch_size=cfg.DETECTOR.BATCH_SIZE, shuffle=True)
    get_iter = lambda epoch, index: (epoch - 1) * len(data_loader) + index
    logger(cfg, args)
    start_index = int(args.patch.split('/')[-1].split('_')[0]) if args.patch is not None else 1
    detector_attacker.init_universal_patch(args.patch)

    losses = []
    save_tensor(detector_attacker.universal_patch, save_name + '.png', args.save_path)
    vlogger = None
    if not args.debugging:
        vlogger = VisualBoard(name=args.board_name, start_iter=start_index, new_process=args.new_process)
        detector_attacker.vlogger = vlogger
    loss_array = []
    for epoch in range(start_index, cfg.ATTACKER.MAX_EPOCH+1):
        et0 = time.time()
        ep_loss = 0
        for index, img_tensor_batch in enumerate(tqdm(data_loader, desc=f'Epoch {epoch}')):
            now_step = get_iter(epoch, index)
            if vlogger: vlogger(epoch, now_step)
            img_tensor_batch = img_tensor_batch.to(detector_attacker.device)
            all_preds = detector_attacker.detect_bbox(img_tensor_batch)
            # get position of adversarial patches
            target_nums = detector_attacker.get_patch_pos_batch(all_preds)
            if sum(target_nums) == 0: continue

            loss = detector_attacker.attack(img_tensor_batch, args.attack_method)
            ep_loss += loss

            if epoch % 10 == 0:
                # patch_name = f'{epoch}_{save_name}'
                patch_name = f'{save_name}' + '.png'
                save_tensor(detector_attacker.universal_patch, patch_name, args.save_path)

        et1 = time.time()
        ep_loss /= len(data_loader)
        loss_array.append(ep_loss)
        if vlogger:
            vlogger.write_ep_loss(ep_loss)
            vlogger.write_scalar(et1-et0, 'misc/ep time')

    np.save(os.path.join(args.save_path, save_name + '-loss.npy'), loss_array)


if __name__ == '__main__':
    from tools.parser import ConfigParser
    from attack.attacker import UniversalAttacker
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, help='fine-tune from a pre-trained patch', default=None)
    parser.add_argument('-a', '--augment_data', action='store_true')
    parser.add_argument('-m', '--attack_method', type=str, default='sequential')
    parser.add_argument('-cfg', '--cfg', type=str, default='test.yaml')
    parser.add_argument('-n', '--board_name', type=str, default=None)
    parser.add_argument('-re', '--random_erase', action='store_true', default=False)
    parser.add_argument('-s', '--save_path', type=str, default='./results/inria')
    parser.add_argument('-d', '--debugging', action='store_true')
    parser.add_argument('-np', '--new_process', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_patch_name = args.cfg.split('/')[-1].split('.')[0] if args.board_name is None else args.board_name
    args.cfg = './configs/' + args.cfg

    print('-------------------------Training-------------------------')
    print('                       device : ', device)
    print('                          cfg :', args.cfg)

    cfg = ConfigParser(args.cfg)
    detector_attacker = UniversalAttacker(cfg, device)
    cfg.show_class_label(cfg.attack_list)
    attack(cfg, detector_attacker, save_patch_name, args)