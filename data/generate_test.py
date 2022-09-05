'''

split data for testing
'''
import os
import numpy as np
import random
import shutil
from tqdm import tqdm

def check(path, rebuild=False):
    if rebuild and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


datasets = ['coco', 'INRIAPerson']
source_map = {'coco': ['train/train2017', 'val/val2017'], 'INRIAPerson': ['Test', 'Train']}

targets = ['ground-truth', 'yolov3', 'yolov3-tiny', 'yolov4', 'yolov4-tiny', 'yolov5', 'faster_rcnn', 'ssd']
targets = ['ground-truth']
dataset = datasets[0]
sources = source_map[dataset]
label_dir = 'labels'
for source in sources:
    for target in targets:
        label_path = f'./{dataset}/{source}/{label_dir}/{target}-labels'
        save_dir = f'./{dataset}/{source}/{label_dir}/{target}-rescale-labels'
        check(save_dir)
        names = os.listdir(label_path)
        for name in tqdm(names):
            tmp = []
            with open(os.path.join(label_path, name), 'r') as f:
                context = f.readlines()
                for con in context:
                    cls, x1, y1, x2, y2 = con.replace('\n', '').split(' ')
                    x1, y1, x2, y2 = np.array([x1, y1, x2, y2], dtype=float) * 416
                    tmp.append(' '.join([cls, str(x1), str(y1), str(x2), str(y2)]))
            res = '\n'.join(tmp)
            with open(os.path.join(save_dir, name), 'w') as f:
                f.write(res)
# check(save_dir, rebuild=True)
# check(save_label, rebuild=True)
#
# # def split_data(data_root, save_dir, file_num):
# img_names = [os.path.join(data_root, i) for i in os.listdir(data_root)]
# if len(img_names) > file_num:
#     # print()
#     random.shuffle(img_names)
#     img_names = img_names[:file_num]
#
# for name in tqdm(img_names):
#     # save = name.replace('train', 'test')
#     cmd = 'cp ' + name + ' coco/test/test2017/'
#     # print(cmd)
#     os.system(cmd)
#     label = name.replace('train2017', 'labels').replace('.jpg', '.txt')
#     assert os.path.exists(label), f'Error, file {label} not exist!'
#
#     cmd = 'cp ' + label + ' coco/test/labels/'
#     os.system(cmd)
    # break
# print('written')