import argparse
import cv2
import json

import os
from tqdm import tqdm


class ConvertCOCOToYOLO:

    """
    Takes in the path to COCO annotations and outputs YOLO annotations in multiple .txt files.
    COCO annotation are to be JSON formart as follows:

        "annotations":{
            "area":2304645,
            "id":1,
            "image_id":10,
            "category_id":4,
            "bbox":[
                0::704
                1:620
                2:1401
                3:1645
            ]
        }
        
    """

    def __init__(self, img_folder, json_path, save_path, name_file):
        self.img_folder = img_folder
        self.json_path = json_path
        self.save_path = save_path
        self.names = load_class_names(name_file)

        os.makedirs(save_path, exist_ok=True)

        self.sep = os.sep

    def get_img_shape(self, img_path):
        img = cv2.imread(img_path)
        assert img is not None, 'Image is None!'
        return img.shape

    def convert_labels(self, img_path, x1, y1, x2, y2):
        """
        Definition: Parses label files to extract label and bounding box
        coordinates. Converts (x1, y1, x1, y2) KITTI format to
        (x, y, width, height) normalized YOLO format.
        """

        def sorting(l1, l2):
            if l1 > l2:
                lmax, lmin = l1, l2
            else:
                lmax, lmin = l2, l1
            return lmax, lmin

        size = self.get_img_shape(img_path)
        xmax, xmin = sorting(x1, x2)
        ymax, ymin = sorting(y1, y2)
        # dw = 1./ size[1]
        # dh = 1./ size[0]
        # x = (xmin + xmax)/2.0
        # y = (ymin + ymax)/2.0
        # w = xmax - xmin
        # h = ymax - ymin
        # x = x*dw
        # w = w*dw
        # y = y*dh
        # h = h*dh
        xmin /= size[1]
        xmax /= size[1]
        ymin /= size[0]
        ymax /= size[0]
        return (xmin, ymin, xmax, ymax)

    def convert(self, annotation_key='annotations', img_id='image_id', cat_id='category_id', bbox_name='bbox'):
        # Enter directory to read JSON file
        data = json.load(open(self.json_path))
        
        check_set = set()

        # Retrieve data
        for key in tqdm(data[annotation_key]):
            # print(key)
            # Get required data
            image_id = f'{key[img_id]}'
            category_id = key[cat_id] - 1
            bbox = key[bbox_name]

            # Retrieve image.
            image_id = ('%12d' % int(image_id)).replace(' ', '0')
            image_path = f'{self.img_folder}{self.sep}{image_id}.jpg'


            # Convert the data: bbox [x, y, w, h] to bbox [x1, y1, x2, y2]
            kitti_bbox = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
            yolo_bbox = self.convert_labels(image_path, kitti_bbox[0], kitti_bbox[1], kitti_bbox[2], kitti_bbox[3])


            # Prepare for export
            label_name = self.names[category_id].replace(' ', '')
            # img = cv2.imread(image_path)
            # print(list(yolo_bbox)+[int(category_id)])
            # plot_boxes_cv2(img, [list(yolo_bbox)+[1.00, int(category_id)]], self.names, savename='/home/chenziyan/work/BaseDetectionAttack/data/test/'+image_id+'.jpg')
            # assert 1 == 0, 'breaking'


            filename = f'{self.save_path}{self.sep}{image_id}.txt'
            content = f"{label_name} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n"

            if label_name[0] == '-':
                content = ''
            # Export 
            if image_id in check_set:
                open_type = 'a'
            else:
                check_set.add(image_id)
                open_type = 'w'

            with open(filename, open_type) as file:
                file.write(content)
            # break
        self.check_empty_label()

    def check_empty_label(self):
        ims = os.listdir(self.img_folder)
        for im_name in tqdm(ims):
            label_name = im_name.split('.')[0] + '.txt'
            label_path = os.path.join(self.save_path, label_name)
            if not os.path.exists(label_path):
                f = open(label_path, 'w')
                f.close()
                print('Empty object: ', label_path)

# To run in as a class
if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from tools.parser import load_class_names
    from tools.det_utils import plot_boxes_cv2

    # print(sys.path)
    target = 'val'
    postfix = '2017'
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_folder', type=str, default=f'./coco/{target}/{target}{postfix}')
    parser.add_argument('-n', '--name_file', type=str, default=f'../configs/namefiles/coco-stuff.names')
    parser.add_argument('-j', '--json_path', type=str, default=f'./coco/instances_{target}{postfix}.json')
    parser.add_argument('-s', '--save_path', type=str, default=f'./coco/{target}/{target}{postfix}_labels')
    args = parser.parse_args()
    util = ConvertCOCOToYOLO(args.img_folder, args.json_path, args.save_path, args.name_file)
    util.convert()
    # util.check_empty_label()
