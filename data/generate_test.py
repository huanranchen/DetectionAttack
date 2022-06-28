'''

split data for testing
'''
import os
import random
import shutil
from tqdm import tqdm

label_path = './coco/train/labels'
data_root = './coco/train/train2017'

save_dir = './coco/test/test2017'
save_label = './coco/test/labels'
file_num = 20000

def check(path, rebuild):
    if rebuild and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

check(save_dir, rebuild=True)
check(save_label, rebuild=True)

# def split_data(data_root, save_dir, file_num):
img_names = [os.path.join(data_root, i) for i in os.listdir(data_root)]
if len(img_names) > file_num:
    # print()
    random.shuffle(img_names)
    img_names = img_names[:file_num]

for name in tqdm(img_names):
    # save = name.replace('train', 'test')
    cmd = 'cp ' + name + ' coco/test/test2017/'
    # print(cmd)
    os.system(cmd)
    label = name.replace('train2017', 'labels').replace('.jpg', '.txt')
    assert os.path.exists(label), f'Error, file {label} not exist!'

    cmd = 'cp ' + label + ' coco/test/labels/'
    os.system(cmd)
    # break
# print('written')