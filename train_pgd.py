import torch
import os
import numpy as np
from tqdm import tqdm
from detector_lab.utils import inter_nms

import time
from tools.draw import VisualBoard
from tools.loader import dataLoader
from tools import save_tensor

def modelDDP(detector_attacker, args):
    for ind, detector in enumerate(detector_attacker.detectors):
        detector_attacker.detectors[ind] = torch.nn.parallel.DistributedDataParallel(detector,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)


def logger(cfg, args, attack_confs_thresh=""):
    import time
    localtime = time.asctime(time.localtime(time.time()))
    print("                        time : ", localtime)
    print('--------------------------DETECTOR----------------------------')
    print("             Attacking model :", cfg.DETECTOR.NAME)
    print('                 CONF_THRESH :', cfg.DETECTOR.CONF_THRESH)
    print('                  IOU_THRESH :', cfg.DETECTOR.IOU_THRESH)
    print('                  Input size :', cfg.DETECTOR.INPUT_SIZE)
    print('                  Batch size :', cfg.DETECTOR.BATCH_SIZE)
    print('               Self-ensemble :', cfg.DETECTOR.PERTURB.GATE)

    print('--------------------------ATTACKER---------------------------')
    print('               Attack method : ', args.attack_method)
    print('                   Loss func : ', cfg.ATTACKER.LOSS_FUNC)
    print("         Attack confs thresh : ", attack_confs_thresh)
    print("                  Patch size : ",
          '['+str(cfg.ATTACKER.PATCH_ATTACK.HEIGHT)+', '+str(cfg.ATTACKER.PATCH_ATTACK.WIDTH)+']')
    print('               Attack method : ', cfg.ATTACKER.METHOD)
    print('             To Augment data : ', cfg.DATA.AUGMENT)
    print('                   Step size : ', cfg.ATTACKER.STEP_SIZE)
    print('------------------------------------------------------------')


def attack(cfg, detector_attacker, save_name, args=None, save_step=5000):
    detector_attacker.gates = {'jitter': False, 'median_pool': False, 'rotate': True, 'shift': False, 'p9_scale': True}
    data_loader = dataLoader(cfg.DATA.TRAIN.IMG_DIR,
                             input_size=cfg.DETECTOR.INPUT_SIZE, is_augment=args.augment_data,
                             batch_size=cfg.DETECTOR.BATCH_SIZE, shuffle=True)
    get_iter = lambda epoch, index: (epoch - 1) * len(data_loader) + index
    logger(cfg, args)
    epoch_save_mode = False if len(data_loader) > save_step else True
    start_index = int(args.patch.split('/')[-1].split('_')[0]) if args.patch is not None else 1
    detector_attacker.init_universal_patch(args.patch)

    losses = []
    save_tensor(detector_attacker.universal_patch, f'{start_index - 1}_{save_name}', args.save_path + '/patch/')
    vlogger = None
    if not args.debugging:
        vlogger = VisualBoard(name=args.board_name, start_iter=start_index)
        detector_attacker.vlogger = vlogger
    for epoch in range(start_index, cfg.ATTACKER.MAX_ITERS+1):
        et0 = time.time()
        for index, img_tensor_batch in enumerate(tqdm(data_loader, desc=f'Epoch {epoch}')):
            now_step = get_iter(epoch, index)
            if vlogger: vlogger(epoch, now_step)
            img_tensor_batch = img_tensor_batch.to(detector_attacker.device)
            all_preds = detector_attacker.detect_bbox(img_tensor_batch)
            # get position of adversarial patches
            target_nums = detector_attacker.get_patch_pos_batch(all_preds)
            if sum(target_nums) == 0: continue

            loss = detector_attacker.attack(img_tensor_batch, args.attack_method)
            if vlogger: vlogger.write_scalar(loss, 'loss/iter loss')
            # the patch will be saved in every 5000 images
            if (epoch_save_mode and index == 1 and epoch % 10 == 0) or (not epoch_save_mode and now_step % save_step == 0):
                prefix = epoch if epoch_save_mode else int(now_step / 5000)
                patch_name = f'{prefix}_{save_name}'
                save_tensor(detector_attacker.universal_patch, patch_name, args.save_path + '/patch/')

        et1 = time.time()
        if vlogger: vlogger.write_scalar(et1-et0, 'misc/ep time')
    np.save(args.save_path+'/losses.npy', np.array(losses))


if __name__ == '__main__':
    from tools.parser import ConfigParser
    from attackAPI import UniversalDetectorAttacker
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, help='fine-tune from a pre-trained patch', default=None)
    parser.add_argument('-a', '--augment_data', action='store_true')
    parser.add_argument('-m', '--attack_method', type=str, default='sequential')
    parser.add_argument('-cfg', '--cfg', type=str, default='test.yaml')
    parser.add_argument('-n', '--board_name', type=str, default=None)
    parser.add_argument('-s', '--save_path', type=str, default='./results/inria')
    parser.add_argument('-d', '--debugging', action='store_true')
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
    attack(cfg, detector_attacker, save_patch_name, args)