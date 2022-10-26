import os
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class eva_args:
    def __init__(self, patch, cfg,
                 label_path=None, data_root=None,
                 save=os.path.join(PROJECT_DIR, '/data/inria/'),
                 stimulate_uint8_loss=False, gen_labels=True, test_origin=False,
                 save_imgs=False, test_gt=False, detectors=None,
                 eva_class=None, quiet=False):
        self.patch = patch
        self.cfg = cfg
        self.save = save
        self.lavel_path = os.path.join(PROJECT_DIR, cfg.DATA.TEST.LAB_DIR) if label_path is None else label_path
        self.data_root = os.path.join(PROJECT_DIR, cfg.DATA.TEST.IMG_DIR) if data_root is None else data_root
        self.stimulate_uint8_loss = stimulate_uint8_loss
        self.gen_labels = gen_labels
        self.test_origin = test_origin
        self.save_imgs = save_imgs
        self.test_gt = test_gt
        self.detectors = cfg.DETECTOR.NAME if detectors is None else detectors
        self.eva_class = cfg.ATTACKER.ATTACK_CLASS if eva_class is None else eva_class
        self.quiet = quiet