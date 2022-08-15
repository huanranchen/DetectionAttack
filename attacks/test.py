import torch


class attacker:
    def __init__(self, loss_function, model, norm='L_infty', epsilons=0.05, max_iters=50, step_size=0.01, class_id=0, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), momentum=0.5):
        self.loss_fn = loss_function

    def init_epsilon(self, detector):
        pass

    def serial_non_targeted_attack(self, ori_tensor_batch, detector_attacker, detector, optimizer):
        # print('============================================')
        adv_tensor_batch, p = detector_attacker.apply_universal_patch(ori_tensor_batch, detector, is_normalize=True)
        # print('requires_grad: ', adv_tensor_batch.requires_grad)
        # interative attack
        # self.optim.zero_grad()
        preds = detector.detect_test(adv_tensor_batch)
        preds = torch.clamp(preds, min=0)
        # print(preds.grad_fn)
        # preds = preds[preds > 0]
        tmp = torch.zeros(preds.shape)
        if preds.is_cuda:
            tmp = tmp.cuda()
        loss = torch.nn.MSELoss()(preds, tmp)
        del p
        return loss