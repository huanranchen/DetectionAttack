import os

# coco_labels_name = ["background", "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
#                     "boat",
#                     "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird",
#                     "cat", "dog", "horse",
#                     "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
#                     "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
#                     "kite", "baseball bat",
#                     "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass",
#                     "cup", "fork", "knife",
#                     "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
#                     "pizza",
#                     "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window",
#                     "desk",
#                     "toilet", "door", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
#                     "oven",
#                     "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear",
#                     "hair drier",
#                     "toothbrush", "hair brush"]
#
# target = coco_labels_name
# with open('../configs/namefiles/coco-91.names', 'w') as f:
#     f.write('\n'.join(target))

import shutil
import os

import numpy as np


def remove_file(old_path, new_path):
    print(old_path)
    print(new_path)
    filelist = os.listdir(old_path) #列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    print(filelist)
    for file in filelist:
        src = os.path.join(old_path, file)
        dst = os.path.join(new_path, file)
        print('src:', src)
        print('dst:', dst)
        shutil.move(src, dst)

def check(query, refer, args=None):
    def filter(target):
        tmp = []
        for t in target:
            cls = t.split(' ')[0]
            # print(int(float(cls)))
            if 'person' in cls or cls[0] == '0': tmp.append(t)
        return tmp
    total = 0
    qsn = 0
    rsn = 0
    for q, r in zip(query, refer):
        with open(q, 'r') as f:
            qs = f.readlines()
        with open(r, 'r') as f:
            rs = f.readlines()


        qs = filter(qs)
        rs = filter(rs)
        qsn += len(qs)
        rsn += len(rs)
        if len(qs) != len(rs):
            total += np.absolute(len(qs)-len(rs))
            print(q, len(qs), len(rs))
            print(qs)
            print(rs)

    print('diff: ', total, args.query, 'query: ', qsn, '; ', args.refer, 'refer: ', rsn)

def check_person(file_path):
    with open(file_path, 'r') as f:
        content = f.readlines()
        for bbox in content:
            bbox = bbox.split(' ')
            if bbox[0] == 'person':
                return True
    return False


if __name__ == '__main__':
    import argparse
    from tqdm import tqdm
    import shutil
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--refer', default='yolov3')
    parser.add_argument('-q', '--query', type=str, default=None)
    args = parser.parse_args()

    model = 'ground-truth'
    models = ['ground-truth', 'yolov3', 'yolov3-tiny', 'yolov4', 'yolov4-tiny', 'yolov5', 'faster_rcnn', 'ssd']
    target = '/train/train2017/'

    source = './coco'
    save = './coco_person'

    label_postfix = 'labels/'+model+'-rescale-labels'
    source_label_fp = source + target + label_postfix
    save_label_fp = save + target + label_postfix
    source_im_fp = source + target + 'pos'
    save_im_fp = save + target + 'pos'
    os.makedirs(save_label_fp, exist_ok=True)
    os.makedirs(save_im_fp, exist_ok=True)

    all_f = os.listdir(source_label_fp)
    person_fs = 0
    for label_f in tqdm(all_f):
        im_f = label_f.replace('.txt', '.jpg')

        label_fp = os.path.join(source_label_fp, label_f)
        if check_person(label_fp):
            # print('have person: ', label_f)
            shutil.copyfile(os.path.join(source_im_fp, im_f), os.path.join(save_im_fp, im_f))
            shutil.copyfile(label_fp, os.path.join(save_label_fp, label_f))
            person_fs += 1

        # if person_fs > 10:
        #     break
