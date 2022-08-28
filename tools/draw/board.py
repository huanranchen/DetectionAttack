from torch.utils.tensorboard import SummaryWriter
import subprocess
import time


class VisualBoard:
    def __init__(self, optimizer=None, name=None, start_iter=0):
        subprocess.Popen(['tensorboard', '--logdir=runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            self.writer = SummaryWriter(f'runs/{time_str}_{name}')
        else:
            self.writer = SummaryWriter()

        self.iter = start_iter
        self.optimizer = optimizer

    def __call__(self, iter):
        self.iter = iter
        self.writer.add_scalar('misc/learning_rate', self.optimizer.param_groups[0]["lr"], self.iter)

    def write_scalar(self, scalar, name):
        self.writer.add_scalar(name, scalar.detach().cpu().numpy(), self.iter)

    def write_tensor(self, im, name):
        try:
            im = im.detach().cpu()
        except:
            pass
        self.writer.add_image('attack/'+name, im, self.iter)

    def write_cv2(self, im, name):
        im = im.transpose((2, 0, 1))
        print(im.shape)
        self.writer.add_image(f'attack/{name}', im, self.iter)

    def write_ep_loss(self, ep_loss):
        self.writer.add_scalar('total_loss/ep_loss', ep_loss.detach().cpu().numpy(), self.iter)

    def write_loss(self, loss, det_loss, tv_loss):
        self.writer.add_scalar('total_loss/loss', loss.detach().cpu().numpy(), self.iter)
        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), self.iter)
        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), self.iter)


