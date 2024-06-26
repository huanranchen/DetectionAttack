import torch
import numpy as np
import os
import multiprocessing
from tools.det_utils import plot_boxes_cv2
from tools import FormatConverter
from detlib.utils import init_detectors
from scripts.dict import get_attack_method, loss_dict
from tools import DataTransformer, pad_lab
from attack.uap import PatchManager, PatchRandomApplier
from tools.det_utils import inter_nms


class UniversalAttacker(object):
    """An attacker agent to coordinate the detect & base attack methods for universal attacks."""

    def __init__(self, cfg, device, model_distribute=False):
        self.cfg = cfg
        self.device = device
        self.max_boxes = 15
        self.patch_boxes = []

        self.class_names = cfg.all_class_names  # class names reference: labels of all the classes
        self.attack_list = cfg.attack_list  # int list: classes index to be attacked, [40, 41, 42, ...]
        self.patch_obj = PatchManager(cfg.ATTACKER.PATCH, device)
        self.vlogger = None

        self.patch_applier = PatchRandomApplier(device, cfg_patch=cfg.ATTACKER.PATCH)
        self.data_transformer = DataTransformer(device, rand_rotate=0)

        self.detectors = init_detectors(cfg_det=cfg.DETECTOR, distribute=model_distribute)
        self.model_distribute = model_distribute

    @property
    def universal_patch(self):
        return self.patch_obj.patch

    def init_attaker(self):
        cfg = self.cfg.ATTACKER
        loss_fn = loss_dict[cfg.LOSS_FUNC]
        self.attacker = get_attack_method(cfg.METHOD)(
            loss_func=loss_fn, norm='L_infty', device=self.device, cfg=cfg, detector_attacker=self)

    def plot_boxes(self, img_tensor, boxes, save_path=None, save_name=None):
        # print(img.dtype, isinstance(img, np.ndarray))
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            save_name = os.path.join(save_path, save_name)
        img = FormatConverter.tensor2numpy_cv2(img_tensor.cpu().detach())
        plot_box = plot_boxes_cv2(img, boxes.cpu().detach().numpy(), self.class_names,
                                  savename=save_name)
        return plot_box

    def init_universal_patch(self, patch_file=None):
        self.patch_obj.init(patch_file)
        # self.universal_patch = self.patch_obj.patch

    def filter_bbox(self, preds, target_cls=None):
        # FIXME: To be a more universal op fn
        if len(preds) == 0: return preds
        # if cls_array is None: cls_array = preds[:, -1]
        # filt = [cls in self.cfg.attack_list for cls in cls_array]
        # preds = preds[filt]
        target_cls = self.cfg.attack_cls if target_cls is None else target_cls
        return preds[preds[:, -1] == target_cls]

    def get_patch_pos_batch(self, all_preds):
        # get all bboxs of setted target. If none target bbox is got, return has_target=False
        self.all_preds = all_preds
        batch_boxes = None
        target_nums = []
        for i_batch, preds in enumerate(all_preds):
            if len(preds) == 0:
                preds = torch.cuda.FloatTensor([[0, 0, 0, 0, 0, 0]])
            preds = self.filter_bbox(preds)
            padded_boxs = pad_lab(preds, self.max_boxes).unsqueeze(0)
            batch_boxes = padded_boxs if batch_boxes is None else torch.vstack((batch_boxes, padded_boxs))
            target_nums.append(len(preds))
        self.all_preds = batch_boxes
        return np.array(target_nums)

    def uap_apply(self, img_tensor, adv_patch=None):
        """
        UAP: universal adversarial patch
        :param img_tensor:
        :param adv_patch:
        :return:
        """
        if adv_patch is None: adv_patch = self.universal_patch
        img_tensor = self.patch_applier(img_tensor, adv_patch, self.all_preds, gates=self.cfg.ATTACKER.PATCH.TRANSFORM)

        # 1st inference: get bbox; 2rd inference: get detections of the adversarial patch
        # if '2' in self.cfg.DATA.AUGMENT: img_tensor = self.data_transformer(img_tensor)

        return img_tensor

    def merge_batch(self, all_preds, preds):
        if all_preds is None:
            return preds
        for i, (all_pred, pred) in enumerate(zip(all_preds, preds)):
            if pred.shape[0]:
                pred = pred.to(all_pred.device)
                all_preds[i] = torch.cat((all_pred, pred), dim=0)
                continue
            all_preds[i] = all_pred if all_pred.shape[0] else pred
        return all_preds

    def detect_bbox(self, img_batch, detectors=None):
        if detectors is None:
            detectors = self.detectors

        all_preds = None
        for detector in detectors:
            preds = detector(img_batch.to(detector.device))['bbox_array']
            all_preds = self.merge_batch(all_preds, preds)

        # nms among detectors
        if len(detectors) > 1: all_preds = inter_nms(all_preds)
        return all_preds

    def attack(self, img_tensor_batch, mode='sequential'):
        '''
        given batch input, return loss, and optimize patch
        '''
        detectors_loss = []
        self.attacker.begin_attack()
        if mode == 'optim' or mode == 'sequential':
            if self.model_distribute and False:
                # TODO: check whether parallel can be a good approximation of second derivative?
                pool = multiprocessing.Pool(processes=len(self.detectors))
                for detector in self.detectors:
                    detectors_loss.append(pool.apply_async(func=self.attacker.non_targeted_attack,
                                                           args=(img_tensor_batch, detector)))
                pool.close()
                pool.join()
            else:
                for detector in self.detectors:
                    loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
                    detectors_loss.append(loss)
        elif mode == 'parallel':
            detectors_loss = self.parallel_attack(img_tensor_batch)
        self.attacker.end_attack()
        return torch.tensor(detectors_loss).mean()

    def parallel_attack(self, img_tensor_batch):
        detectors_loss = []
        patch_updates = torch.zeros(self.universal_patch.shape).to(self.device)
        for detector in self.detectors:
            patch_tmp, loss = self.attacker.non_targeted_attack(img_tensor_batch, detector)
            patch_update = patch_tmp - self.universal_patch
            patch_updates += patch_update
            detectors_loss.append(loss)
        self.patch_obj.update_((self.universal_patch + patch_updates / len(self.detectors)).detach_())
        return detectors_loss
