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

if __name__ == '__main__':
    import argparse
    target = 'Test'
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--refer', default='yolov3')
    parser.add_argument('-q', '--query', type=str, default=None)
    args = parser.parse_args()
    if args.query is None:
        args.query = args.refer
    # remove_file(r"./inra/gap/aug/inria0/all_data", r"./inria/aug/in-dataset-gap/aug/all_data")

    query_dir = f'./INRIAPerson/{target}/labels/{args.query}-labels'
    query_dir = f'./INRIAPerson/{target}/labels/yolov2-rescale-labels'
    # query_dir = f'./INRIAPerson/{target}/labels/bak/{args.query}-labels'

    refer_dir = f'./INRIAPerson/{target}/yolo-labels-natural/{args.refer}-labels'
    refer_dir = f'./INRIAPerson/{target}/labels/origin/yolov2-rescale-labels'
    qs = [os.path.join(query_dir, d) for d in os.listdir(query_dir)]
    rs = [os.path.join(refer_dir, d) for d in os.listdir(refer_dir)]
    check(qs, rs, args)
