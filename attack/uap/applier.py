"""

This is not used since tons of tensors takes huge GPU memory
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .transformer import PatchTransformer


class PatchRandomApplier(nn.Module):
    # apply patch
    def __init__(self, device, rotate_angle=20, rand_loc_rate=0.1, scale_rate=0.2):
        """

        :param rotate_angle: random rotate angle range from [-rotate_angle, rotate_angle]
        :param rand_loc_rate: random shift rate (of patch) range
        :param scale_rate: patch size / bbox size
        """
        super().__init__()
        self.patch_transformer = PatchTransformer(device, rotate_angle, rand_loc_rate, scale_rate).to(device)
        self.device = device

    def list2tensor(self, list_batch, max_len=10):
        """This made this class an agent, the outside funcs don't have to care about the processing
        of the bbox format, and the PatchTransformer only need to process the uniformed bbox torch tensor batch.

        :param bboxes:
        :return:
        """
        bboxes_tensor_batch = None
        for i, bboxes_list in enumerate(list_batch):
            # print(f'batch {i}', len(bboxes_list))
            if type(bboxes_list) is np.ndarray or type(bboxes_list) is list:
                bboxes_list = torch.cuda.FloatTensor(bboxes_list)
            print(bboxes_list.size(0))
            if bboxes_list.size(0) == 0:
                padded_lab = torch.zeros((max_len, 6)).unsqueeze(0).to(self.device)
            else:
                bboxes_list = bboxes_list[:max_len + 1]
                pad_size = max_len - len(bboxes_list)
                # print(bboxes_list, pad_size)
                padded_lab = F.pad(bboxes_list, (0, 0, 0, pad_size), value=0).unsqueeze(0)

            if bboxes_tensor_batch is None:
                bboxes_tensor_batch = padded_lab
            else:
                bboxes_tensor_batch = torch.cat((bboxes_tensor_batch, padded_lab))
        # print('list2tensor :', bboxes_tensor_batch.shape)
        return bboxes_tensor_batch

    def forward(self, img_batch, adv_patch, bboxes_batch, gates):
        """ This func to process the bboxes list of mini-batch into uniform torch.tensor and
        apply the patch into the img batch. Every patch stickers will be randomly transformed
        by given transform range before being attached.

        :param img_batch:
        :param adv_patch:
        :param bboxes_batch: bbox [batch_size, [N*6]]
        :return:
        """
        # print(img_batch.size, adv_patch.size)
        gates = patch_aug_gates(gates)
        patch_ori_size = adv_patch.size(-1)
        batch_size = img_batch.size(0)
        pad_size = (img_batch.size(-1) - adv_patch.size(-1)) / 2
        padding = nn.ConstantPad2d((int(pad_size + 0.5), int(pad_size), int(pad_size + 0.5), int(pad_size)), 0)  # (LRTB)

        # if isinstance(bboxes_batch, list):
        #     bboxes_batch = self.list2tensor(bboxes_batch)
        lab_len = bboxes_batch.size(1)
        # --------------Median pool degradation & Random jitter---------------------
        adv_batch = adv_patch.unsqueeze(0)
        if gates['median_pool']:
            adv_batch = self.patch_transformer.median_pooler(adv_batch[0])
        adv_batch = adv_batch.expand(batch_size, lab_len, -1, -1, -1) # [batch_size, lab_len, 3, N, N]
        if gates['jitter']:
            adv_batch = self.patch_transformer.random_jitter(adv_batch)
        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)
        if gates['cutout']:
            adv_batch = self.patch_transformer.random_erase(adv_batch)
        adv_batch = padding(adv_batch)

        # transform by gates
        adv_batch_t = self.patch_transformer(adv_batch, bboxes_batch, patch_ori_size,
                                             rand_rotate_gate=gates['rotate'],
                                             rand_shift_gate=gates['shift'],
                                             p9_scale=gates['p9_scale'], rdrop=gates['rdrop'])

        adv_img_batch = PatchApplier.forward(img_batch, adv_batch_t)
        # print('Patch apply out: ', adv_img_batch.shape)
        return adv_img_batch


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """
    @staticmethod
    def forward(img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            img_batch = torch.where((adv == 0), img_batch, adv)
        return img_batch



def patch_aug_gates(aug_list):
    gates = {'jitter': False, 'median_pool': False, 'rotate': False, 'shift': False, 'p9_scale': False, 'rdrop': False, 'cutout': False}
    for aug in aug_list:
        gates[aug] = True
    return gates