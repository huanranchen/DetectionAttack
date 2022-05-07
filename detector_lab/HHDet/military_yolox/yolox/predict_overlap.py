import numpy as np
import cv2
import tqdm
import math
import os
from PIL import Image
import torch
import scipy.misc
import matplotlib.pyplot as plt
import time
from yolo import YOLO


def get_file_names(data_dir, file_type):
    result_dir = []
    result_name = []
    for maindir, subdir, file_name_list in os.walk(data_dir):
        for filename in file_name_list:
            apath = maindir+'/'+filename
            ext = apath.split('.')[-1]
            if ext in file_type:
                result_dir.append(apath)
                result_name.append(filename)
            else:
                pass
    return result_dir, result_name

def cutimg_overlap(in_dir, out_dir, file_type, cutshape, overlap_factor, out_type):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    data_dir_list, _ = get_file_names(in_dir, file_type)
    count = 0
    print('Cut begining for ', str(len(data_dir_list)), ' images.....')
    for each_dir in tqdm.tqdm(data_dir_list):
        img = cv2.imread(each_dir)
        print(img.shape)
        factor1 = int(math.ceil(img.shape[0] / overlap_factor))
        factor2 = int(math.ceil(img.shape[1] / overlap_factor))
        for i in range(factor2):
            for ii in range(factor1):
                start_x = ii * cutshape - ii * overlap_factor
                end_x = (ii + 1) * (cutshape) - ii * overlap_factor
                start_y = i * cutshape - i * overlap_factor
                end_y = (i + 1) * cutshape - i * overlap_factor
                if end_x > img.shape[0]:
                    start_x = img.shape[0] - cutshape
                    end_x = img.shape[0]
                if end_y > img.shape[1]:
                    start_y = img.shape[1] - cutshape
                    end_y = img.shape[1]
                img_temp = img[start_x:end_x, start_y:end_y, :]
                out_dir_images = out_dir + '/' + each_dir.split('/')[-1].split('.')[0] \
                                 + '_' + str(start_x) + '_' + str(end_x) + '_' + str(start_y) + '_' + str(
                    end_y) + '.' + out_type
                #######进行小图检测
                image = Image.fromarray(img_temp)
                r_image, result_place, result_class, result_confidence, coordinates_list = yolo.detect_image_xml(image)

                #######进行小图检测
                cv2.imwrite(out_dir_images, img_temp)
                if end_x == img.shape[0] and end_y == img.shape[1]:
                    break


if __name__ == '__main__':
    ##### cut
    yolo = YOLO()
    data_dir = 'E:\YOLOX\det-net\det-net\overlap_test'
    out_dir = 'E:\YOLOX\det-net\det-net\overlap_test'
    file_type = ['jpg','png']
    out_type = 'png'
    cutshape = 640
    overlap_factor = 100

    cutimg_overlap(data_dir, out_dir, file_type, cutshape, overlap_factor, out_type)
