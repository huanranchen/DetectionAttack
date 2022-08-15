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


def logger(cfg, args, attack_confs_thresh):
    print('-------------------DETECTOR---------------------')
    print("Attacking model              :", cfg.DETECTOR.NAME)
    print('Conf thresh                  :', cfg.DETECTOR.CONF_THRESH)
    print('IOU_THRESH                   :', cfg.DETECTOR.IOU_THRESH)
    print('Input size                   :', cfg.DETECTOR.INPUT_SIZE)
    print('Batch size                   :', cfg.DETECTOR.BATCH_SIZE)

    to_perturb = False
    if hasattr(cfg.DETECTOR, 'GRAD_PERTURB'):
        to_perturb = cfg.DETECTOR.GRAD_PERTURB
    print('To perturb(self-ensemble)    :', to_perturb)

    print('-------------------ATTACKER---------------------')
    print('Attack method                : ', args.attack_method)
    print("Attack confs thresh          : ", attack_confs_thresh)
    print("Patch size                   : ", cfg.ATTACKER.PATCH_ATTACK)
    print('Attack method                : ', cfg.ATTACKER.METHOD)
    print('To Augment data              : ', cfg.DATA.AUGMENT)
    print('Step size                    : ', cfg.ATTACKER.STEP_SIZE)


def attack(cfg, data_root, detector_attacker, save_name, args=None):
    from tools.lr_decay import cosine_decay
    attack_confs_thresh = cfg.DETECTOR.CONF_THRESH - 0.2
    logger(cfg, args, attack_confs_thresh)
    save_plot=True
    # detector_attacker.ddp = False
    data_sampler = None
    detector_attacker.init_universal_patch(args.patch)
    detector_attacker.save_patch(args.save_path, f'0_{save_name}')
    # if detector_attacker.ddp:
    #
    #     import torch.distributed as dist
    #     torch.cuda.set_device(args.local_rank)
    #     dist.init_process_group(backend='nccl')
    #     data_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
    #     modelDDP(detector_attacker, args)

    data_loader = dataLoader(data_root, cfg.DETECTOR.INPUT_SIZE, is_augment=cfg.DATA.AUGMENT,
                             batch_size=cfg.DETECTOR.BATCH_SIZE, sampler=data_sampler)

    save_step = 5000
    epoch_save_mode = False if len(data_loader) > save_step else True

    start_index = int(args.patch.split('_')[0].split('/')[-1]) if args.patch is not None else 1
    losses = []
    for epoch in range(start_index, cfg.ATTACKER.MAX_ITERS+1):
        if args.confs_thresh_decay:
            attack_confs_thresh = cosine_decay(epoch-1, lr_max=cfg.DETECTOR.CONF_THRESH, lr_min=0.05)
        print('confs threshold: ', attack_confs_thresh)
        # torch.cuda.empty_cache()
        for index, img_tensor_batch in tqdm(enumerate(data_loader)):
            now_step = index + epoch * len(data_loader)
            all_preds = None
            for detector in detector_attacker.detectors:
                # detector.target = None  # TODO: CONF_POLICY test
                img_tensor_batch = img_tensor_batch.to(detector.device)
                preds, _ = detector.detect_img_batch_get_bbox_conf(img_tensor_batch)
                all_preds = detector_attacker.merge_batch_pred(all_preds, preds)

                detector.target = preds # TODO: CONF_POLICY test

            # nms among detectors
            if len(detector_attacker.detectors) > 1:
                all_preds = inter_nms(all_preds)

            # get position of adversarial patches
            target_nums = detector_attacker.get_patch_pos_batch(all_preds)
            if sum(target_nums) == 0:
                print('no target detected.')
                continue

            if args.attack_method == 'parallel':
                detector_attacker.parallel_attack(img_tensor_batch)
            elif args.attack_method == 'serial':
                loss = detector_attacker.serial_attack(img_tensor_batch, confs_thresh=attack_confs_thresh)

            losses.append(loss)
            if save_plot and index % 10 == 0:
                for detector in detector_attacker.detectors:
                    detector_attacker.imshow_save(img_tensor_batch, os.path.join(args.save_path, detector.name),
                                                  save_name, detectors=[detector])

            # if index == 1:
            #     detector_attacker.save_patch(args.save_path, '1_'+save_name)
            # the patch will be saved in every 5000 images
            if (epoch_save_mode and index == 1 and epoch % 10 == 0) \
                    or (not epoch_save_mode and now_step % save_step == 0):
                prefix = epoch if epoch_save_mode else int(now_step / 5000)
                patch_name = f'{prefix}_{save_name}'
                detector_attacker.save_patch(args.save_path, patch_name)

            if hasattr(cfg.DETECTOR, 'GRAD_PERTURB') and cfg.DETECTOR.GRAD_PERTURB:
                freq = cfg.DETECTOR.PERTURB_FREQ if hasattr(cfg.DETECTOR, 'PERTURB_FREQ') else 1
                print('GRAD_PERTURB: every('.lower(), freq, ' step)')
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
    parser.add_argument('-m', '--attack_method', type=str, default='serial')
    parser.add_argument('-n', '--nesterov', action='store_true')
    parser.add_argument('-cfg', '--cfg', type=str, default='test.yaml')
    parser.add_argument('-s', '--save_path', type=str, default='./results/inria')
    parser.add_argument('-d', '--confs_thresh_decay', action='store_true')
    parser.add_argument('-rk', '--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    args = parser.parse_args()

    print('-----------------------Training----------------------------')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device               : ', device)
    save_patch_name = args.cfg.split('.')[0] + '.png'
    args.cfg = './configs/' + args.cfg
    print('cfg                  :', args.cfg)
    cfg = ConfigParser(args.cfg)
    detector_attacker = UniversalDetectorAttacker(cfg, device)

    cfg.show_class_label(cfg.attack_list)

    data_root = cfg.DATA.TRAIN.IMG_DIR
    img_names = [os.path.join(data_root, i) for i in os.listdir(data_root)]

    import warnings
    warnings.filterwarnings('ignore')
    attack(cfg, data_root, detector_attacker, save_patch_name, args)