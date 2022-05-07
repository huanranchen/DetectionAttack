
import time

import cv2
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

from frcnn import FRCNN

if __name__ == "__main__":
    frcnn = FRCNN()
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    img_names = os.listdir(dir_origin_path)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path  = os.path.join(dir_origin_path, img_name)
            print(image_path)
            img_tensor, image = frcnn.prepare_img(image_path)
            print(image)
            top_boxes, top_conf, top_label = frcnn.detect_image(img_tensor, image)

            r_image = frcnn.draw_boxes(image, top_label, top_boxes, top_conf)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

