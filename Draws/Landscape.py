import os
from .DrawUtils.D2Landscape import D2Landscape
import torch
import cv2
import numpy as np
import sys
sys.path.append('./')
from landscape import get_loss
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

total_image_in_set = 1000  # 训练集，测试集到底有多少image？


class Landscape():
    '''

    '''

    def __init__(self,
                 patches_path: str,
                 output_path: str,
                 configs,
                 mode='3D',
                 iter_each_config=1,
                 one_image = False):
        '''
        :param configs: for one figure, use multiple configs
        :param mode: 3D? Contour?
        :param one_image: only use one image to eval?
        '''
        self.patches_path = patches_path
        self.output_path = output_path
        self.args = configs
        self.mode = mode
        self.iter_each_config = iter_each_config
        self.one_image = one_image

        if not os.path.exists(output_path):
            os.makedirs(output_path)

    def draw(self, total=10):
        '''
        :param total:  how many patches you want to draw?
        '''
        patches = os.listdir(self.patches_path)
        for i, patch in enumerate(patches):
            self._draw_one(patch, self.iter_each_config)
            if i > total:
                break

    def _draw_one(self, patch_name, step_each_point=1):
        figure = plt.figure()
        axes = Axes3D(figure)
        patch = self.read_patch(self.patches_path + patch_name)
        use_image = np.random.randint(0, total_image_in_set) if self.one_image else None
        instance_train = D2Landscape(
            lambda x: get_loss(self.args[0], x, step_each_point, use_which_image=use_image),
            patch,
            mode=self.mode
        )
        coordinate = instance_train.synthesize_coordinates()
        instance_train.draw(axes)
        for i, config in enumerate(self.args):
            if i >= 1:
                use_image = np.random.randint(0, total_image_in_set) if self.one_image else None
                instance_val = D2Landscape(
                    lambda x: get_loss(self.args[i], x, step_each_point, use_which_image=use_image),
                    patch,
                    mode=self.mode
                )
                instance_val.assign_coordinates(*coordinate)
                instance_val.draw(axes)
        plt.show()
        plt.savefig(self.output_path + patch_name + ".png")
        plt.clf()
        plt.close()

    @staticmethod
    def read_patch(patch_file):
        print('reading patch ' + patch_file)
        universal_patch = cv2.imread(patch_file)
        universal_patch = cv2.cvtColor(universal_patch, cv2.COLOR_BGR2RGB)
        universal_patch = np.expand_dims(np.transpose(universal_patch, (2, 0, 1)), 0)
        universal_patch = torch.from_numpy(np.array(universal_patch, dtype='float32') / 255.).cuda()
        return universal_patch

@torch.no_grad()
def train_valid_3dlandscape():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco0.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/data/coco/train/train2017')
    args_train = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco0.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/data/coco/test/test2017')
    args_test = parser.parse_args()

    a = Landscape(patches_path="/home/chenziyan/work/results/coco/partial/patch/",
                  output_path='./Draws/out/',
                  configs=[args_train, args_test],
                  mode='3D',
                  iter_each_config=1)
    a.draw(total=10)

@torch.no_grad()
def multi_model_3dlandscape():
    '''
    通过传入多个config file来实现multi model
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco0.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/data/coco/train/train2017')
    args_train = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco3.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/data/coco/train/train2017')
    args_test = parser.parse_args()

    a = Landscape(patches_path="/home/chenziyan/work/results/coco/partial/patch/",
                  output_path='./Draws/out/',
                  configs=[args_train, args_test],
                  mode='3D',
                  iter_each_config=1)
    a.draw(total=10)

@torch.no_grad()
def multi_image_contourf(num_image=3):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco0.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/data/coco/train/train2017')
    args_train = parser.parse_args()

    a = Landscape(patches_path="/home/chenziyan/work/results/coco/partial/patch/",
                  output_path='./Draws/out/',
                  configs=[args_train] * num_image,
                  mode='Contour',
                  iter_each_config=1,
                  one_image=True)
    a.draw(total=10)

@torch.no_grad()
def multi_model_contourf():
    '''
    通过传入多个config file来实现multi model
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco0.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/data/coco/train/train2017')
    args_train = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--patch', type=str, default='None')
    parser.add_argument('-cfg', '--config_file', type=str, default='./configs/coco3.yaml')
    parser.add_argument('-dr', '--data_root', type=str,
                        default='/home/chenziyan/work/BaseDetectionAttack/data/coco/train/train2017')
    args_test = parser.parse_args()

    a = Landscape(patches_path="/home/chenziyan/work/results/coco/partial/patch/",
                  output_path='./Draws/out/',
                  configs=[args_train, args_test],
                  mode='Contour',
                  iter_each_config=1)
    a.draw(total=10)

if __name__ == '__main__':
    train_valid_3dlandscape()