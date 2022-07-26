from tqdm import tqdm
from detector_lab.utils import inter_nms
from tools.data_loader import read_img_np_batch

from tools.data_loader import detDataSet
from torch.utils.data import DataLoader


def modelDDP(detector_attacker, args):
    for ind, detector in enumerate(detector_attacker.detectors):
        detector_attacker.detectors[ind] = torch.nn.parallel.DistributedDataParallel(detector,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

def attack(cfg, data_root, detector_attacker, save_name, args=None):
    conf_thresh = cfg.DETECTOR.CONF_THRESH
    if not args.const_confs:
        conf_thresh /= 2
    save_plot=True
    # detector_attacker.ddp = False
    data_sampler = None
    detector_attacker.init_universal_patch()
    data_set = detDataSet(data_root, cfg.DETECTOR.INPUT_SIZE)

    # if detector_attacker.ddp:
    #
    #     import torch.distributed as dist
    #     torch.cuda.set_device(args.local_rank)
    #     dist.init_process_group(backend='nccl')
    #     data_sampler = torch.utils.data.distributed.DistributedSampler(data_set)
    #     modelDDP(detector_attacker, args)

    data_loader = DataLoader(data_set, batch_size=cfg.DETECTOR.BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True, sampler=data_sampler)

    for epoch in range(cfg.ATTACKER.MAX_ITERS):
        # torch.cuda.empty_cache()
        for index, img_tensor_batch in tqdm(enumerate(data_loader)):
            all_preds = None
            for detector in detector_attacker.detectors:
                img_tensor_batch = img_tensor_batch.to(detector.device)
                preds, _ = detector.detect_img_batch_get_bbox_conf(img_tensor_batch)
                all_preds = detector_attacker.merge_batch_pred(all_preds, preds)

                detector.target = preds

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
                detector_attacker.serial_attack(img_tensor_batch, confs_thresh=conf_thresh)

            if save_plot:
                for detector in detector_attacker.detectors:
                    detector_attacker.imshow_save(img_tensor_batch, os.path.join(args.save_path, detector.name),
                                                  save_name, detectors=[detector])

            if epoch % 10 == 0:
                patch_name = f'{epoch}_{save_name}'
                print(patch_name)
                detector_attacker.save_patch(args.save_path, patch_name)


if __name__ == '__main__':
    import torch
    import os
    from tools.parser import ConfigParser
    from attackAPI import UniversalDetectorAttacker
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--attack_method', type=str, default='serial')
    parser.add_argument('-cfg', '--cfg', type=str, default='test.yaml')
    parser.add_argument('-s', '--save_path', type=str, default='./results/inria')
    parser.add_argument('-rk', '--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
    parser.add_argument('-cc', '--const_confs', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)
    save_patch_name = args.cfg.split('.')[0] + '.png'
    args.cfg = './configs/' + args.cfg
    cfg = ConfigParser(args.cfg)
    detector_attacker = UniversalDetectorAttacker(cfg, device)

    cfg.show_class_label(cfg.attack_list)

    data_root = cfg.DETECTOR.IMG_DIR
    img_names = [os.path.join(data_root, i) for i in os.listdir(data_root)]

    import warnings
    warnings.filterwarnings('ignore')
    attack(cfg, data_root, detector_attacker, save_patch_name, args)