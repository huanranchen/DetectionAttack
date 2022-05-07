import os
import numpy as np
import cv2
from tqdm import tqdm
import xml.dom.minidom as xmldom


def load_class_names(namesfile):
    class_names = []
    with open(namesfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.rstrip()
        class_names.append(line)
    return class_names

def parse_xml(fn):
    xml_file = xmldom.parse(fn)
    eles = xml_file.documentElement
    print(eles.tagName)
    labels = []
    width = float(eles.getElementsByTagName("width")[0].firstChild.data)
    height = int(eles.getElementsByTagName('height')[0].firstChild.data)
    for ind, xmin in enumerate(eles.getElementsByTagName("xmin")):
        xmin = xmin.firstChild.data
        # print(eles.getElementsByTagName("xmin")[1].firstChild.data)
        xmax = eles.getElementsByTagName("xmax")[ind].firstChild.data
        ymin = eles.getElementsByTagName("ymin")[ind].firstChild.data
        ymax = eles.getElementsByTagName("ymax")[ind].firstChild.data
        name = eles.getElementsByTagName("name")[ind].firstChild.data
        # labels.append([int(xmin), int(ymin), int(xmax), int(ymax)])
        label = [float(xmin)/width, float(ymin)/height, float(xmax)/width, float(ymax)/ height]
        label = [str(i) for i in label]
        label.insert(0, name)
        label = ' '.join(label)
        labels.append(label)

    # print(labels)
    return labels

def transfer(img_path, save_path, save_split, file):
    with open(file) as f:
        lines = f.readlines()

    for line in tqdm(lines):
        info = line.split(' ')
        im_file = info[0].split('/')[-1]
        im_file = im_file.replace('\n', '')

        txt_file = im_file.split('.')[0] + '.txt'
        boxes = info[1:]
        img = cv2.imread(os.path.join(img_path, im_file))
        h, w, _ = img.shape
        s = ''
        for box in boxes:
            tmp = np.array([float(i) for i in box.split(',')])
            tmp[[0, 2]] = tmp[[0, 2]] / w
            tmp[[1, 3]] = tmp[[1, 3]] / h
            tmp = np.r_[tmp[-1], tmp[:-1]]
            tmp[0] = int(list(tmp)[0])
            tmp = [str(i) for i in tmp]
            # print(tmp)
            s += save_split.join(tmp) + '\n'
        with open(os.path.join(save_path, txt_file), 'w') as f:
            f.write(s)
        # break


if __name__ == "__main__":
    gt_path = './military_data/Annotations'
    img_path = './military_data/JPEGImages'
    save_path = './military_data/AnnotationLabels'
    name_file = '../configs/namefiles/military.names'
    save_split = ' '

    os.makedirs(save_path, exist_ok=True)

    names = load_class_names(name_file)

    print(names)
    cnt = 0
    files = os.listdir(gt_path)
    for file in files:
        cnt += 1
        print(file)
        labels = parse_xml(os.path.join(gt_path, file))
        file = file.split('.')[0] + '.txt'
        with open(os.path.join(save_path, file), 'w') as f:
            f.write('\n'.join(labels))
        # break
    print(cnt)
    # file = './2007_train.txt'
    # file = './2007_val.txt'
    # transfer(img_path, save_path, save_split, file)