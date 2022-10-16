import os
import torch
import torchvision.transforms
from .base import BaseAttacker
from torch import nn
from PIL import Image


class ResidualBlock(nn.Module):
    def __init__(self, *args):
        super(ResidualBlock, self).__init__()
        self.model = nn.ModuleList([*args])

    def forward(self, x):
        for m in self.model:
            x = m(x)
        return x


class BasicDownSample(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True):
        super(BasicDownSample, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU() if relu else None,
        )

    def forward(self, x):
        return self.model(x)


class BasicUpSample(nn.Module):
    def __init__(self, in_channels, out_channels, relu=True):
        super(BasicUpSample, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if relu else nn.Identity(),
        )

    def forward(self, x):
        return self.model(x)


class SimpleUNet(nn.Module):
    def __init__(self, nn_config=[6, 15, 30, 50], lr=1e-3):
        super(SimpleUNet, self).__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in range(len(nn_config) - 1):
            self.down.append(BasicDownSample(nn_config[i], nn_config[i + 1], relu=True))
        for i in range(len(nn_config) - 1, 0, -1):
            self.up.append(BasicUpSample(nn_config[i], nn_config[i - 1], relu=True))
        self.up.append(nn.Conv2d(nn_config[0], 3, kernel_size=1, stride=1, padding=0))  # down的地方多了一个映射为3channel的

        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, amsgrad=True)

    def forward(self, x):
        '''
        N. C. H. D
        '''
        forward_cache = [x]
        for m in self.down:
            x = m(x)
            forward_cache.append(x)

        forward_cache.pop(-1)
        forward_cache.reverse()
        for m in self.up[:-1]:
            x = m(x)
            if len(forward_cache) != 0:
                now = forward_cache.pop(0)
                if now.shape[3] == x.shape[3] - 1:
                    x = x[:, :, :-1, :-1]
                x += now

        m = self.up[-1]
        x = m(x)
        return x

    def update(self):
        self.optimizer.step()

    def zero_gradient(self):
        self.optimizer.zero_grad()


class NNattacker(BaseAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty',
                 ):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)
        self.get_patch()
        self.model = SimpleUNet(lr=cfg.START_LEARNING_RATE)

    @torch.no_grad()
    def get_patch(self, path='./patches'):
        to_tensor = torchvision.transforms.ToTensor()
        patch = []
        patches_dir = os.listdir(path)
        for patch_dir in patches_dir:
            now_patch = Image.open(os.path.join(path, patch_dir))
            now_patch = to_tensor(now_patch)
            patch.append(now_patch)

        self.patch = torch.cat(patch, dim=0)  # N*C, H, D

    def generate_patch(self):
        x = self.model(self.patch.unsqueeze(0))
        return x.squeeze()

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []
        for iter in range(self.iter_step):
            if iter > 0:
                ori_tensor_batch = ori_tensor_batch.clone()

            self.detector_attacker.patch_obj.patch = self.generate_patch()

            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            # detect adv img batch to get bbox and obj confs
            bboxes, confs, cls_array = detector(adv_tensor_batch).values()

            if hasattr(self.cfg, 'class_specify'):
                attack_cls = int(self.cfg.ATTACK_CLASS)
                confs = torch.cat(
                    ([conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
            elif hasattr(self.cfg, 'topx_conf'):
                confs = torch.sort(confs, dim=-1, descending=True)[0][:, :self.cfg.topx_conf]
                confs = torch.mean(confs, dim=-1)
            else:
                confs = confs.max(dim=-1, keepdim=True)[0]

            detector.zero_grad()
            self.model.zero_gradient()
            loss_dict = self.attack_loss(confs=confs)
            loss = loss_dict['loss']
            loss.backward()
            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.model.update()
        # print(adv_tensor_batch, bboxes, loss_dict)
        # update training statistics on tensorboard
        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

    def attack_loss(self, confs):
        '''
        SAME with P9 and optim attacker
        '''
        loss = self.loss_fn(confs=confs, patch=self.detector_attacker.universal_patch[0])
        if 'obj-tv' in self.cfg.LOSS_FUNC:
            tv_loss, obj_loss = loss.values()
            tv_loss = torch.max(self.cfg.tv_eta * tv_loss, torch.tensor(0.1).to(self.device))
            obj_loss = obj_loss * self.cfg.obj_eta
            loss = tv_loss + obj_loss
        elif self.cfg.LOSS_FUNC == 'obj':
            loss = loss['obj_loss'] * self.cfg.obj_eta
            tv_loss = torch.cuda.FloatTensor([0])
        out = {'loss': loss, 'det_loss': obj_loss, 'tv_loss': tv_loss}
        return out
