import torch
import os
import numpy as np
from tqdm import tqdm
from detector_lab.utils import inter_nms

from tools.data_loader import dataLoader


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


def attack(cfg, data_root, detector_attacker, save_name, args=None):
    from tools.lr_decay import cosine_decay

    attack_confs_thresh = cfg.DETECTOR.CONF_THRESH - 0.2
    if hasattr(cfg.DETECTOR, 'ATTACK_CONF_THRESH'):
        attack_confs_thresh = cfg.DETECTOR.ATTACK_CONF_THRESH

    data_sampler = None
    data_loader = dataLoader(data_root, cfg.DETECTOR.INPUT_SIZE, is_augment=cfg.DATA.AUGMENT==1,
                             batch_size=cfg.DETECTOR.BATCH_SIZE, sampler=data_sampler)
    logger(cfg, args, attack_confs_thresh)
    save_step = 5000
    save_plot=True
    # detector_attacker.ddp = False

    epoch_save_mode = False if len(data_loader) > save_step else True
    start_index = int(args.patch.split('/')[-1].split('_')[0]) if args.patch is not None else 1
    detector_attacker.init_universal_patch(args.patch)
    detector_attacker.save_patch(args.save_path, f'{str(start_index)}_{save_name}')
    # if detector_attacker.ddp:
    #
    #     import torch.distributed as dist
    #     torch.cuda.set_device(args.local_rank)
    #     dist.init_process_group(backend='nccl')
    #     data_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
    #     modelDDP(detector_attacker, args)

    losses = []
    for epoch in range(start_index, cfg.ATTACKER.MAX_ITERS+1):
        if args.confs_thresh_decay:
            attack_confs_thresh = cosine_decay(epoch-1, lr_max=cfg.DETECTOR.CONF_THRESH, lr_min=0.05)
        print('confs threshold: ', attack_confs_thresh)
        for index, img_tensor_batch in enumerate(tqdm(data_loader, desc=f'Epoch {epoch}')):
            now_step = index + epoch * len(data_loader)
            img_tensor_batch = img_tensor_batch.to(detector_attacker.device)
            all_preds = detector_attacker.detect_bbox(img_tensor_batch)
            # get position of adversarial patches
            target_nums = detector_attacker.get_patch_pos_batch(all_preds)
            if sum(target_nums) == 0:
                print('no target detected.')
                continue

            detector_attacker.attack(img_tensor_batch, args.attack_method, confs_thresh=attack_confs_thresh)

            if save_plot and index % 10 == 0:
                # for detector-specific dir name
                for detector in detector_attacker.detectors:
                    detector_attacker.adv_detect_save(img_tensor_batch,
                                                      os.path.join(args.save_path, detector.name),
                                                      save_name,
                                                      detectors=[detector])

            # the patch will be saved in every 5000 images
            if (epoch_save_mode and index == 1 and epoch % 10 == 0) \
                    or (not epoch_save_mode and now_step % save_step == 0):
                prefix = epoch if epoch_save_mode else int(now_step / 5000)
                patch_name = f'{prefix}_{save_name}'
                detector_attacker.save_patch(args.save_path, patch_name)

            if cfg.DETECTOR.PERTURB.GATE == 'grad_descend':
                try:
                    freq = cfg.DETECTOR.PERTURB.GRAD_DESCEND.PERTURB_FREQ
                except Exception as e:
                    print(e, 'From grad perturb: Resetting model reset freq...')
                    freq = 1

                print('GRAD_PERTURB: every(', freq, ' step)')
                if now_step and now_step % cfg.DETECTOR.RESET_FREQ == 0:
                    print(now_step, ' : resetting the model')
                    detector.reset_model()
                elif index % freq == 0:
                    print(now_step, ': perturbing')
                    detector.perturb()

    np.save(args.save_path+'/losses.npy', np.array(losses))


if __name__ == '__main__':
    from tools.parser import ConfigParser
    from attackAPI import UniversalDetectorAttacker
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, help='fine-tune from a pre-trained patch', default=None)
    parser.add_argument('-m', '--attack_method', type=str, default='sequential')
    parser.add_argument('-n', '--nesterov', action='store_true')
    parser.add_argument('-cfg', '--cfg', type=str, default='test.yaml')
    parser.add_argument('-s', '--save_path', type=str, default='./results/inria')
    parser.add_argument('-d', '--confs_thresh_decay', action='store_true')
    parser.add_argument('-rk', '--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
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