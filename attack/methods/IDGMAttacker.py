import torch
from .optim import OptimAttacker
from torch.optim import Optimizer, Adam


def cosine_similarity(x: list):
    '''
    input a list of tensor with same shape. return the mean cosine_similarity
    '''
    x = torch.stack(x, dim=0)
    N = x.shape[0]
    x = x.reshape(N, -1)

    norm = torch.norm(x, p=2, dim=1)
    x /= norm.reshape(-1, 1)  # N, D
    similarity = x @ x.T  # N, N
    mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=0).to(torch.bool)  # 只取上三角
    similarity = similarity[mask]
    return torch.mean(similarity).item()


class LazyAttacker(OptimAttacker):
    threshold = 0.75
    '''
    小于阈值，我就不更新了。只有大于阈值，我才会正常更新
    '''

    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty',
                 out_optimizer=Adam, ksi=0.05):
        # FIXME: delete outer optimizer. Use normal optimization.
        super().__init__(device, cfg, loss_func, detector_attacker, norm=norm)
        self.ksi = ksi
        self.out_optimizer = out_optimizer

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer
        if self.out_optimizer is not None:
            print(f'set outer optimizer is {self.out_optimizer}')
            print('-' * 100)
            self.out_optimizer = self.out_optimizer([self.optimizer.param_groups[0]['params'][0]], self.ksi)
        if self.detector_attacker.vlogger is not None:
            self.detector_attacker.vlogger.optimizer = self.out_optimizer

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []
        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            adv_tensor_batch = adv_tensor_batch.to(detector.device)
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
            self.grad_record.append(self.optimizer.param_groups[0]['params'][0].grad.clone())
            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update()
        # print(adv_tensor_batch, bboxes, loss_dict)
        # update training statistics on tensorboard
        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

    @torch.no_grad()
    def begin_attack(self):
        self.original_patch = self.optimizer.param_groups[0]['params'][0].detach().clone()
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self):
        '''
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        '''
        grad_similarity = cosine_similarity(self.grad_record)
        self.detector_attacker.vlogger.write_scalar(grad_similarity, 'grad_similarity')
        if grad_similarity < self.threshold:
            patch = self.optimizer.param_groups[0]['params'][0]
            patch.mul_(0)
            patch.add_(self.original_patch)
        else:
            patch = self.optimizer.param_groups[0]['params'][0]
            if self.out_optimizer is None:
                patch.mul_(self.ksi)
                patch.add_((1 - self.ksi) * self.original_patch)
                self.original_patch = None
            else:
                fake_grad = - self.ksi * (patch - self.original_patch)
                self.out_optimizer.zero_grad()
                patch.mul_(0)
                patch.add_(self.original_patch)
                patch.grad = fake_grad
                self.out_optimizer.step()

        del self.grad_record


class FishAttacker(OptimAttacker):
    '''
    use fish technique to approximate second derivatives.
    '''

    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty',
                 out_optimizer=Adam, ksi=0.05):
        super().__init__(device, cfg, loss_func, detector_attacker, norm=norm)
        self.ksi = ksi
        self.out_optimizer = out_optimizer

    @property
    def param_groups(self):
        return self.out_optimizer.param_groups

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer
        if self.out_optimizer is not None:
            print(f'set outer optimizer is {self.out_optimizer}')
            print('-' * 100)
            self.out_optimizer = self.out_optimizer([self.optimizer.param_groups[0]['params'][0]], self.ksi)
        if self.detector_attacker.vlogger is not None:
            self.detector_attacker.vlogger.optimizer = self.out_optimizer

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []
        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            adv_tensor_batch = adv_tensor_batch.to(detector.device)
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
            self.grad_record.append(self.optimizer.param_groups[0]['params'][0].grad.clone())
            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update()
        # print(adv_tensor_batch, bboxes, loss_dict)
        # update training statistics on tensorboard
        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

    @torch.no_grad()
    def begin_attack(self):
        self.original_patch = self.optimizer.param_groups[0]['params'][0].detach().clone()
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self):
        '''
        theta: original_patch
        theta_hat: now patch in optimizer
        theta = theta + ksi*(theta_hat - theta), so:
        theta =(1-ksi )theta + ksi* theta_hat
        '''
        patch = self.optimizer.param_groups[0]['params'][0]
        if self.out_optimizer is None:
            patch.mul_(self.ksi)
            patch.add_((1 - self.ksi) * self.original_patch)
            self.original_patch = None
        else:
            fake_grad = - self.ksi * (patch - self.original_patch)
            self.out_optimizer.zero_grad()
            patch.mul_(0)
            patch.add_(self.original_patch)
            patch.grad = fake_grad
            self.out_optimizer.step()

        grad_similarity = cosine_similarity(self.grad_record)
        self.detector_attacker.vlogger.write_scalar(grad_similarity, 'grad_similarity')
        del self.grad_record


class SmoothFishAttacker(OptimAttacker):
    '''
    use fish technique to approximate second derivatives.
    attention!!!!! twice the time!!!!!!!
    '''

    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty', out_optimizer=None):
        super().__init__(device, cfg, loss_func, detector_attacker, norm=norm)
        self.out_optimizer = out_optimizer

    @property
    def param_groups(self):
        return self.out_optimizer.param_groups

    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer
        if self.out_optimizer is not None:
            print(f'set outer optimizer is {self.out_optimizer}')
            print('-' * 100)
            self.out_optimizer = self.out_optimizer([self.optimizer.param_groups[0]['params'][0]], self.ksi)
        if self.detector_attacker.vlogger is not None:
            self.detector_attacker.vlogger.optimizer = self.out_optimizer

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []
        for iter in range(self.iter_step):
            ############################# for theta_hat ##############################
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            adv_tensor_batch = adv_tensor_batch.to(detector.device)
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
            self.theta_hat_grad_record.append(self.optimizer.param_groups[0]['params'][0].grad.clone())
            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update()

            ################################ theta #########################################
            # first step: change the now_theta to theta, backup the now_theta as theta_hat
            # attention! use inplace operation to avoid some other problems!!!!!!
            with torch.no_grad():
                now_theta = self.optimizer.param_groups[0]['params'][0].detach().clone()
                self.optimizer.param_groups[0]['params'][0].mul_(0)
                self.optimizer.param_groups[0]['params'][0].add_(self.original_patch)

            # second step: do the forward pass and get the gradient
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            adv_tensor_batch = adv_tensor_batch.to(detector.device)
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
            loss_dict = self.attack_loss(confs=confs)
            loss = loss_dict['loss']
            loss.backward()
            self.theta_grad_record.append(self.optimizer.param_groups[0]['params'][0].grad.clone())

            # third step: get the theta_hat back
            with torch.no_grad():
                self.optimizer.param_groups[0]['params'][0].mul_(0)
                self.optimizer.param_groups[0]['params'][0].add_(now_theta)

        # print(adv_tensor_batch, bboxes, loss_dict)
        # update training statistics on tensorboard
        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

    @torch.no_grad()
    def begin_attack(self):
        self.original_patch = self.optimizer.param_groups[0]['params'][0].detach().clone()
        self.theta_hat_grad_record = []  # this is useless.
        self.theta_grad_record = []

    @torch.no_grad()
    def end_attack(self, ksi=0.05, gamma=5):
        '''
        grad_mean = mean(d theta)
        grad_sm = alpha*num_detectors*grad_mean + gamma*( (theta_hat - theta) - alpha*num_detectors*grad_mean )

        :param gamma:
        if gamma is 1, this is fish. if gamma is 0, this is ERM(parallel attack)
        '''
        alpha = self.optimizer.param_groups[0]['lr']
        num_detectors = len(self.theta_grad_record)
        grad_mean = torch.mean(torch.stack(self.theta_grad_record), dim=0)
        theta_hat = self.optimizer.param_groups[0]['params'][0].detach().clone()
        grad_sm = alpha * num_detectors * grad_mean + \
                  gamma * ((theta_hat - self.original_patch) - alpha * num_detectors * grad_mean)
        # backup to original patch
        self.optimizer.param_groups[0]['params'][0].mul_(0)
        self.optimizer.param_groups[0]['params'][0].add_(self.original_patch)
        # outer loop update. ksi*grad_sm
        if self.out_optimizer is None:
            self.optimizer.param_groups[0]['params'][0].add_(ksi * grad_sm)
        else:
            self.out_optimizer.zero_grad()
            self.optimizer.param_groups[0]['params'][0].grad = - ksi * grad_sm
            self.out_optimizer.step()

        # record in tensorboard
        grad_similarity = cosine_similarity(self.theta_grad_record)
        self.detector_attacker.vlogger.write_scalar(grad_similarity, 'grad_similarity')
        del self.theta_grad_record
        del self.theta_hat_grad_record


class OptimAttackerWithRecord(OptimAttacker):
    def __init__(self, device, cfg, loss_func, detector_attacker, norm='L_infty'):
        super().__init__(device, cfg, loss_func, detector_attacker, norm=norm)

    def non_targeted_attack(self, ori_tensor_batch, detector):
        losses = []
        for iter in range(self.iter_step):
            if iter > 0: ori_tensor_batch = ori_tensor_batch.clone()
            adv_tensor_batch = self.detector_attacker.uap_apply(ori_tensor_batch)
            adv_tensor_batch = adv_tensor_batch.to(detector.device)
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
            loss_dict = self.attack_loss(confs=confs)
            loss = loss_dict['loss']
            loss.backward()
            self.grad_record.append(self.optimizer.param_groups[0]['params'][0].grad.clone())
            losses.append(float(loss))

            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update()
        # print(adv_tensor_batch, bboxes, loss_dict)
        # update training statistics on tensorboard
        self.logger(detector, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

    @torch.no_grad()
    def begin_attack(self):
        self.grad_record = []

    @torch.no_grad()
    def end_attack(self):
        '''
        set to tensorboard
        '''
        grad_similarity = cosine_similarity(self.grad_record)
        self.detector_attacker.vlogger.write_scalar(grad_similarity, 'grad_similarity')
        del self.grad_record
