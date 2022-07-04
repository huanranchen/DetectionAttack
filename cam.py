import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

from detector_lab.HHDet.utils import init_detector
from tools.data_loader import read_img_np_batch
from tools.det_utils import plot_boxes_cv2
from tools.parser import load_class_names

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

model = init_detector('YOLOV4')
print(model.device)
target_layers = [model.module_list[-3]]
print(target_layers)
# model = resnet50(pretrained=True)
# target_layers = [model.layer4[-1]]
class_file = './configs/namefiles/coco.names'
cls = load_class_names(class_file, trim=False)#[str(i) for i in list(np.arange(0, 80, 1))]
img_dir = './data/coco/06-01/7_99000_coco0/YOLOV3/imgs'
# img_dir = './data/INRIAPerson/Test/pos/'
# img_dir = './data/coco/train/train2017'
imgs =  os.listdir(img_dir)

for index, img_name in tqdm(enumerate(imgs)):
    img_path = os.path.join(img_dir, img_name)
    img = read_img_np_batch([img_path], input_size=(416, 416))
    rgb_img = np.transpose(img[0], (1, 2, 0))
    tensor, _ = model.normalize(rgb_img) # Create an input tensor image for your model..

    # test detection
    # print(tensor.shape)
    img_tensor = model.init_img_batch(img).to(model.device)
    preds, _ = model.detect_img_batch_get_bbox_conf(img_tensor)
    img_numpy, img_numpy_int8 = model.unnormalize(img_tensor[0])
    plot = plot_boxes_cv2(img_numpy_int8, preds[0], cls, savename=None)
    # cv2.imwrite(f'test.png', plot)
    # break

    cam = EigenCAM(model.detector, target_layers, use_cuda=True)
    # cam = GradCAM(model=model.detector, target_layers=target_layers, use_cuda=True)

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    targets = [ClassifierOutputTarget(80)]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = cam(tensor, targets=targets)[0, ...]

    # In this example grayscale_cam has only one image in the batch:
    # grayscale_cam = grayscale_cam[0, :]
    # plot_img =
    plot_img = (plot / 255.).astype(np.float32)
    # plot_img = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
    visualization = show_cam_on_image(plot_img, grayscale_cam, use_rgb=True)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'./data/cam/{img_name}.png', visualization)
    # break