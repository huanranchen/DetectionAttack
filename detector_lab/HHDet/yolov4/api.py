import sys
import os
import argparse


from .Pytorch_YOLOv4.tool.utils import *
from .Pytorch_YOLOv4.tool.torch_utils import *
from .Pytorch_YOLOv4.tool.darknet2pytorch import Darknet

import torch
from torch.autograd import Variable
import numpy as np
import cv2


class HHYolov4:
    def __init__(self, name, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        # name
        self.name = name

        # detector
        self.detector = None

        # device (cuda or cpu)
        self.device = device

        
    
    def load(self, cfg_file,  weightfile, data_config_path):

        self.detector = Darknet(cfg_file)

        # data config
        self.num_classes = self.detector.num_classes
        if self.num_classes == 20:
            namesfile = os.path.join(data_config_path, 'data/voc.names')
        elif self.num_classes == 80:
            namesfile = os.path.join(data_config_path, 'data/coco.names')
        else:
            namesfile = os.path.join(data_config_path, 'data/x.names')
        self.namesfile = load_class_names(namesfile)

        # post_processing method | input (img, conf_thresh, nms_thresh, output) | return bboxes_batch
        self.post_processing = post_processing

        self.detector.load_weights(weightfile)
        self.detector.to(self.device)

    def init_img_batch(self, img_numpy_batch):
        img_tensor = Variable(torch.from_numpy(img_numpy_batch).float().div(255.0).to(self.device))
        return img_tensor

    def prepare_img(self, img_path=None, img_cv2=None, input_shape=(416, 416)):
        """prepare a image from img path or cv2 img

        Args:
            img_path (str, optional): the path of input image. Defaults to None.
            img_cv2 (np.numpy, optional): the cv2 type image. Defaults to None.
            input_shape (tuple, optional): the size to resize. Defaults to 416.

        Raises:
            Exception: if no input image, raise the exception

        Returns:
            tuple: [torch.Tensor, np.numpy], the torch tensor of input image, the cv2 type image of input
        """

        if img_path:
            img_cv2 = cv2.imread(img_path)
        elif img_cv2:
            img_cv2 = img_cv2
        else:
            raise Exception('no input image!')
        
        sized = cv2.resize(img_cv2, input_shape)
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).to(self.device)
        img_tensor = Variable(img_tensor) # , requires_grad=True
        return img_tensor, img_cv2

    def zero_grad(self):
        self.detector.zero_grad()

    def normalize(self, image_data):
        # init numpy into tensor & preprocessing (if mean-std(or other) normalization needed)
        # image_data: np array [h, w, c]
        # print('normalize: ', image_data.shape)
        image_data = np.array(image_data, dtype='float32') / 255.
        image_data = np.expand_dims(np.transpose(image_data, (2, 0, 1)), 0)
        img_tensor = torch.from_numpy(image_data).to(self.device)
        img_tensor.requires_grad = True
        return img_tensor, image_data

    def normalize_tensor(self, img_tensor):
        # for if mean-std(or other) normalization needed
        tensor_data = img_tensor.clone()
        tensor_data.requires_grad = True
        return tensor_data

    def unnormalize(self, img_tensor):
        # img_tensor: tensor [1, c, h, w]
        img_numpy = img_tensor.squeeze(0).cpu().detach().numpy()
        img_numpy = np.transpose(img_numpy, (1, 2, 0))
        img_numpy *= 255
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        img_numpy_int8 = img_numpy.astype('uint8')
        return img_numpy, img_numpy_int8

    def detect_cv2_show(self, imgfile, savename='predictions.jpg'):
        """detect a image with yolov4 and draw the bounding boxes

        Args:
            imgfile ([str]): [the path of image to be detected]

        Returns:
            boxes[0] ([list]): [detected boxes]
        """
        img = cv2.imread(imgfile)
        sized = cv2.resize(img, (self.detector.width, self.detector.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        boxes = do_detect(self.detector, sized, 0.4, 0.6, self.device==torch.device("cuda:0"))
        self.plot_boxes_cv2(imgfile, boxes[0], savename)
    
    def plot_boxes_cv2(self, imgfile, boxes, savename=None):
        """[summary]

        Args:
            imgfile ([cv2.image]): [the path of image to be detected]
            boxes ([type]): [detected boxes]
            savename ([str], optional): [save image name]. Defaults to None.

        Returns:
            [cv2.image]: [cv2 type image with drawn boxes]
        """
        class_names = self.namesfile
        img = cv2.imread(imgfile)
        img = np.copy(img)
        colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

        def get_color(c, x, max_val):
            ratio = float(x) / max_val * 5
            i = int(math.floor(ratio))
            j = int(math.ceil(ratio))
            ratio = ratio - i
            r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
            return int(r * 255)

        width = img.shape[1]
        height = img.shape[0]
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            bbox_thick = int(0.6 * (height + width) / 600)
            rgb = (255, 0, 0)
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                print('%s: %f' % (class_names[cls_id], cls_conf))
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                red = get_color(2, offset, classes)
                green = get_color(1, offset, classes)
                blue = get_color(0, offset, classes)
                if color is None:
                    rgb = (red, green, blue)
                msg = str(class_names[cls_id])+" "+str(round(cls_conf,3))
                t_size = cv2.getTextSize(msg, 0, 0.7, thickness=bbox_thick // 2)[0]
                c1, c2 = (x1,y1), (x2, y2)
                c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
                
                cv2.rectangle(img, (x1,y1), (np.int(c3[0]), np.int(c3[1])), rgb, -1)
                img = cv2.putText(img, msg, (c1[0], np.int(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,0,0), bbox_thick//2,lineType=cv2.LINE_AA)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, bbox_thick)
        if savename:
            print("save plot results to %s" % savename)
            cv2.imwrite(savename, img)
        return img

    def detect_img_tensor_get_bbox_conf(self, input_img, ori_img_cv2, conf_thresh=0.5, nms_thresh=0.4):
        self.detector.eval()
        output = self.detector(input_img)
        # [batch, num, 1, 4] e.g., [1, 22743, 1, 4]
        box_array = output[0] 
        # [batch, num, num_classes] e.g.,[1, 22743, 80]
        confs = output[1]
        preds = self.post_processing(input_img, conf_thresh, nms_thresh, output)[0]
        return preds, confs

    def detect_img_batch_get_bbox_conf(self, input_img, conf_thresh=0.5, nms_thresh=0.4):
        self.detector.eval()
        output = self.detector(input_img)
        # [batch, num, 1, 4] e.g., [1, 22743, 1, 4]
        # box_array = output[0]
        # [batch, num, num_classes] e.g.,[1, 22743, 80]
        confs = output[1]
        # confs = confs.view(confs.shape[0], -1)
        preds = self.post_processing(input_img, conf_thresh, nms_thresh, output)
        # print('v4 h: ', confs.shape, confs.requires_grad)
        return preds, confs
    
    def temp_loss(self, confs):
        return torch.nn.MSELoss()(confs, torch.ones(confs.shape).to(self.device))



if __name__ == '__main__':
    hhyolov4 = HHYolov4()
    hhyolov4.load('/home/huanghao/BaseDetectionAttack/detector_lab/pytorch-YOLOv4/weight/yolov4.weights')
    # # hhyolov4.detect_cv2_show('/home/huanghao/BaseDetectionAttack/detector_lab/pytorch-YOLOv4/data/giraffe.jpg')
    # img = cv2.imread('/home/huanghao/BaseDetectionAttack/detector_lab/pytorch-YOLOv4/data/giraffe.jpg')
    # sized = cv2.resize(img, (hhyolov4.detector.width, hhyolov4.detector.height))
    # sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    # img_tensor = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0).cuda()
    # img_tensor = Variable(img_tensor, requires_grad=True)
    # box_array, confs = hhyolov4.detect_img_tensor_get_bbox_conf(img_tensor)
    # loss = hhyolov4.temp_loss(confs)
    # loss.backward()
    # print(img_tensor.grad)




