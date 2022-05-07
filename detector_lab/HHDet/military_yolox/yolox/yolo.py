import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from PIL import Image
from nets.yolo import YoloBody
from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_outputs, non_max_suppression

class YOLO(object):
    _defaults = {
        "model_path"        : 'logs/ep1155-loss1.474-val_loss1.452.pth',
        "classes_path"      : 'model_data/car_classes.txt',
        "input_shape"       : [640,640],
        "phi"               : 'x',
        "confidence"        : 0.5,
        "nms_iou"           : 0.3,
        "letterbox_image"   : True,
        "cuda"              : True,
    }
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, device=None, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.class_names, self.num_classes  = get_classes(self.classes_path)

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

    def generate(self):
        self.net    = YoloBody(self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()

        print('{} model, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    def read_img(self, img_path):
        return Image.open(img_path)

    def prepare_img(self, image, img_size=None):
        # image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        img_tensor = torch.from_numpy(image_data).to(self.device)

        return image, img_tensor

    def detect_image(self, image, img_tensor):
        # image, img_tensor = self.prepare_img(img_path)
        image_shape = np.array(np.shape(image)[0:2])
        with torch.no_grad():
            outputs = self.net(img_tensor)
            outputs = decode_outputs(outputs, self.input_shape)
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
            if results[0] is None: 
                return image
            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        return top_boxes, top_label, top_conf

    def draw_boxes(self, image, top_label, top_boxes, top_conf):
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def detect_image_xml(self, image):#由detect_image更改而来，负责生成xml，同时自适应尺寸
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #image_data = resize_image(image, (image_shape[1], image_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            #outputs = decode_outputs(outputs, image_shape)
            results = non_max_suppression(outputs, self.num_classes, self.input_shape,
                                          image_shape, self.letterbox_image, conf_thres=self.confidence,
                                          nms_thres=self.nms_iou)
            #results = non_max_suppression(outputs, self.num_classes, image_shape,
            #                              image_shape, self.letterbox_image, conf_thres=self.confidence,
            #                              nms_thres=self.nms_iou)
            if results[0] is None:
                return image
            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]

        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(image_shape), 1))
        #######
        coordinates_list = []
        classIDs = []
        result_place = torch.zeros(top_boxes.shape[0], 4).numpy().astype('int32')
        result_confidence = torch.zeros(top_boxes.shape[0], 1).numpy().astype('float32')
        result_class = torch.zeros(top_boxes.shape[0], 1).numpy().astype('str')
        #######
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            #######
            result_class[i,0] = predicted_class
            result_confidence[i, 0] = score
            result_place[i,0] = top
            result_place[i,1] = left
            result_place[i,2] = bottom
            result_place[i,3] = right
            coordinates_list.append([left,top, right, bottom, int(c)])
            #######
            # if predicted_class == 'truck':
            #     predicted_class = str('卡车')
            # if predicted_class == 'panzer':
            #     predicted_class = str('装甲车')
            # if predicted_class == 'SUV':
            #     predicted_class = str('越野车')
            # if predicted_class == 'cam_tar':
            #     predicted_class = str('伪装目标')
            # if predicted_class == 'cam_net':
            #     predicted_class = str('伪装网')
            # if predicted_class == 'tank':
            #     predicted_class = str('坦克')

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            #print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(3*thickness):  # 包围框变粗
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image,result_place,result_class,result_confidence,coordinates_list,image_shape

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])

        image       = cvtColor(image)

        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)

        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)

            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                  
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                outputs = self.net(images)
                outputs = decode_outputs(outputs, self.input_shape)
                results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = decode_outputs(outputs, self.input_shape)
            results = non_max_suppression(outputs, self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return

if __name__ == "__main__":
    import os
    from tqdm import tqdm

    dir_origin_path = r"img/"
    img_names = os.listdir(dir_origin_path)

    yolo = YOLO()
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(
                ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(dir_origin_path, img_name)
            image, img_tensor = yolo.prepare_img(image_path)
            top_boxes, top_label, top_conf = yolo.detect_image(image, img_tensor)

            # show detection results
            dir_save_path = r"img_out/"
            r_image = yolo.draw_boxes(image, top_label, top_boxes, top_conf)
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            r_image.save(os.path.join(dir_save_path, img_name))
