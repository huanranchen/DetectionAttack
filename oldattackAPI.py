import numpy as np
import yaml
import math
import copy

from detector_lab.HHDet.yolov3.api import HHYolov3
from detector_lab.HHDet.yolov4.api import HHYolov4
from detector_lab.HHDet.military_faster_rcnn.api import FRCNN as Military_FasterRCNN
from detector_lab.HHDet.military_yolox.api import YOLO as Military_YOLOX

from attacks.bim import LinfBIMAttack
from attacks.mim import LinfMIMAttack
from attacks.pgd import LinfPGDAttack

from losses import temp_attack_loss
from BaseDetectionAttack.tools.utils import obj, plot_boxes_cv2


attacker_dict = {
    "bim": LinfBIMAttack,
    "mim": LinfMIMAttack,
    "pgd": LinfPGDAttack,
}

def init_detectors(cfg):
    detectors = []
    for detector_name in cfg.DETECTOR.NAME:
        
        if detector_name == "YOLOV3":
            detector = HHYolov3(name="YOLOV3")
            detector.load(detector_config_file='./detector_lab/HHDet/yolov3/PyTorch_YOLOv3/config/yolov3.cfg', detector_weights='./detector_lab/HHDet/yolov3/PyTorch_YOLOv3/weights/yolov3.weights')
        
        if detector_name == "YOLOV4":
            detector = HHYolov4(name="YOLOV4")
            detector.load('./detector_lab/HHDet/yolov4/Pytorch_YOLOv4/cfg/yolov4.cfg', './detector_lab/HHDet/yolov4/Pytorch_YOLOv4/weight/yolov4.weights', './detector_lab/HHDet/yolov4/Pytorch_YOLOv4')

        if detector_name == "Military_FasterRCNN":
            detector = Military_FasterRCNN(name="Military_FasterRCNN")
            detector.load(detector_weights='./detector_lab/HHDet/military_faster_rcnn/faster_rcnn/logs/ep300-loss0.701-val_loss1.057.pth', classes_path='./detector_lab/HHDet/military_faster_rcnn/faster_rcnn/model_data/car_classes.txt')

        if detector_name == "Military_YOLOX":
            detector = Military_YOLOX(name="Military_YOLOX")
            detector.load(detector_weights='./detector_lab/HHDet/military_yolox/yolox/logs/ep1155-loss1.474-val_loss1.452.pth', classes_path='./detector_lab/HHDet/military_yolox/yolox/model_data/car_classes.txt')
 
        detectors.append(detector)

    return detectors

class DetctorAttacker(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.detectors = init_detectors(cfg)
        self.attacker = attacker_dict[cfg.ATTACKER.METHOD](loss_fuction=temp_attack_loss, model=self.detectors, norm='L_infty', 
        epsilons=cfg.ATTACKER.EPSILON, max_iters=cfg.ATTACKER.MAX_ITERS, step_size=cfg.ATTACKER.STEP_SIZE,
        class_id=cfg.ATTACKER.TARGET_CLASS)
        self.patch_boxes = []
    
    def plot_boxes(self, img, boxes, savename=None):
        def load_class_names(namesfile):
            class_names = []
            with open(namesfile, 'r') as fp:
                lines = fp.readlines()
            for line in lines:
                line = line.rstrip()
                class_names.append(line)
            return class_names
        class_names = load_class_names(self.cfg.DETECTOR.CLASS_NAME_FILE)
        plot_boxes_cv2(img, boxes, class_names, savename=savename)
    
    def get_patch_pos(self, preds, img_cv2):
        width = img_cv2.shape[1]
        height = img_cv2.shape[0]
        patch_boxs = []
        for pred in preds:
            x1, y1, x2, y2, conf, id = pred
            if self.cfg.ATTACKER.ATTACK_CLASS == -1 or self.cfg.ATTACKER.ATTACK_CLASS == int(id):
                p_x1 = ((x1 + x2) / 2) - ((math.sqrt(self.cfg.ATTACKER.PATCH_ATTACK.SCALE) * (x2 - x1)) / 2)
                p_x1 = int(p_x1.clip(0, 1) * width)
                p_y1 = ((y1 + y2) / 2) - ((math.sqrt(self.cfg.ATTACKER.PATCH_ATTACK.SCALE) * (y2 - y1)) / 2)
                p_y1 = int(p_y1.clip(0, 1) * height)
                p_x2 = ((x1 + x2) / 2) + ((math.sqrt(self.cfg.ATTACKER.PATCH_ATTACK.SCALE) * (x2 - x1)) / 2)
                p_x2 = int(p_x2.clip(0, 1) * width)
                p_y2 = ((y1 + y2) / 2) + ((math.sqrt(self.cfg.ATTACKER.PATCH_ATTACK.SCALE) * (y2 - y1)) / 2)
                p_y2 = int(p_y2.clip(0, 1) * height)
                patch_boxs.append([p_x1, p_y1, p_x2, p_y2])
        self.patch_boxes += patch_boxs
    
    def init_patches(self):
        for i in range(len(self.patch_boxes)):
            p_x1, p_y1, p_x2, p_y2 = self.patch_boxes[i][:4]
            patch = np.random.randint(low=0, high=255, size=(p_y2 - p_y1, p_x2 - p_x1, 3))
            self.patch_boxes[i].append(patch) # x1, y1, x2, y2, patch

    def apply_patches(self, img_tensor, detector, is_normalize=True):
        # for each patch in the patch_boxes
        for i in range(len(self.patch_boxes)):
            p_x1, p_y1, p_x2, p_y2 = self.patch_boxes[i][:4]
            if is_normalize:
                # init the patches
                patch_tensor, patch_cv2 = detector.normalize(self.patch_boxes[i][-1])
            else:
                # when attacking
                patch_tensor = self.patch_boxes[i][-1].detach()
                patch_tensor.requires_grad = True
            self.patch_boxes[i][-1] = patch_tensor  # x1, y1, x2, y2, patch_tensor
            # print('add: ', img_tensor.shape, patch_tensor.is_leaf)
            img_tensor[0, :, p_y1:p_y2, p_x1:p_x2] = patch_tensor
        return img_tensor
            

    
if __name__ == '__main__':
    cfg = yaml.load(open('./configs/default.yaml'), Loader=yaml.FullLoader)
    cfg = obj(cfg)
    detector_attacker = DetctorAttacker(cfg)
    img_path = './mytest.png'
    for detector in detector_attacker.detectors:
        # data prepare
        img_tensor, img_cv2 = detector.prepare_img(img_path=img_path)
        ori_img_tensor = copy.deepcopy(img_tensor)
        # print('ori:', ori_img_tensor.requires_grad)
        # get object bbox
        preds, detections_with_grad = detector.detect_img_tensor_get_bbox_conf(input_img=img_tensor, ori_img_cv2=img_cv2)
        # print(preds)
        # get position of adversarial patches
        detector_attacker.get_patch_pos(preds, img_cv2)
        # init the adversarial patches
        detector_attacker.init_patches()
        # add patch to the input tensor
        adv_img_tensor = detector_attacker.apply_patches(img_tensor, detector)

        
        adv_img_tensor = detector_attacker.attacker.non_targeted_attack(ori_img_tensor, adv_img_tensor, img_cv2, detector_attacker, detector)

        preds, detections_with_grad = detector.detect_img_tensor_get_bbox_conf(input_img=adv_img_tensor, ori_img_cv2=img_cv2)
        
        img_numpy, img_numpy_int8 = detector.unnormalize(adv_img_tensor)
        # cv2.imwrite('./results/{}_temp.png'.format(detector.name), img_numpy_int8)
    

        # preds, detections_with_grad = detector.detect_img_tensor_get_bbox_conf(input_img=adv_img_tensor, ori_img_cv2=img_cv2)
        # disappear_loss = detector.temp_loss(detections_with_grad)
        # disappear_loss.backward()
        detector_attacker.plot_boxes(img_numpy_int8, preds, savename='./results/{}_results.png'.format(detector.name))
        # for i in range(len(detector_attacker.patch_boxes)):
        #     print(detector_attacker.patch_boxes[i][-1].grad.shape)