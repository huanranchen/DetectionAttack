import torch
from .optim import OptimAttacker


class FishAttacker(OptimAttacker):
    '''
    use fish technique to approximate second derivatives.
    '''

    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(loss_func, norm, cfg, device, detector_attacker)

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []
        for iter in range(self.iter_step):
            if iter > 0:
                ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            # detect adv img batch to get bbox and obj confs
            bboxes, confs, cls_array = detector(adv_tensor_batch).values()

            if hasattr(self.cfg, 'class_specify'):
                attack_cls = int(self.cfg.ATTACK_CLASS)
                confs = torch.cat(
                    ([conf[cls == attack_cls].max(dim=-1, keepdim=True)[0] for conf, cls in zip(confs, cls_array)]))
            elif hasattr(self.cfg, 'topx_conf'):
                # attack top x confidence
                # print(confs.size())
                confs = torch.sort(confs, dim=-1, descending=True)[0][:, :self.cfg.topx_conf]
                confs = torch.mean(confs, dim=-1)
            else:
                # only attack the max confidence
                confs = confs.max(dim=-1, keepdim=True)[0]

            detector.zero_grad()
            # print('confs', confs)
            loss_dict = self.attack_loss(confs=confs)
            loss = loss_dict['loss']
            # print(loss)
            loss.backward()
            # print(self.detector_attacker.patch_obj.patch.grad)
            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update()
        # print(adv_tensor_batch, bboxes, loss_dict)
        # update training statistics on tensorboard
        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

    def begin_attack(self):
        self.original_patch = self.optimizer.param_groups[0]['params'][0].detach().clone()

    def end_attack(self, ksi=0.1):
        '''
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        '''
        patch = self.optimizer.param_groups[0]['params'][0]
        patch.mul_(ksi)
        patch.add_((1 - ksi) * self.original_patch)
        self.original_patch = None
