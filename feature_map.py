import os
import cv2
import numpy as np
from tqdm import tqdm
import torch

from tools import FormatConverter
from detlib.utils import init_detector
from tools.loader import read_img_np_batch
from tools.det_utils import plot_boxes_cv2
from tools.parser import load_class_names
from tools.parser import ConfigParser
from tools.loader import dataLoader
from evaluate import UniversalPatchEvaluator

from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


def draw_cam(model, img_tensor_batch, plot, save_dir, save_name):
    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    targets = [ClassifierOutputTarget(80)]
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    # grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = cam(img_tensor_batch, targets=targets)[0, ...]

    # In this example grayscale_cam has only one image in the batch:
    # grayscale_cam = grayscale_cam[0, :]
    plot_img = (plot / 255.).astype(np.float32)
    # plot_img = cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB)
    visualization = show_cam_on_image(plot_img, grayscale_cam, use_rgb=True)
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(save_dir, save_name), visualization)


def detect(attacker, img_tensor_batch):
    all_preds = attacker.detect_bbox(img_tensor_batch)
    attacker.get_patch_pos_batch(all_preds)
    return all_preds


def draw_detection(attacker, img_tensor_batch, save_dir, save_name=None):
    preds = detect(attacker, img_tensor_batch)

    savename = None
    if save_name is not None:
        savename = os.path.join(save_dir, save_name)

    if img_tensor_batch.device != torch.device('cpu'):
        img_tensor_batch = img_tensor_batch.cpu()
        preds = preds[0].cpu().detach()
    img = FormatConverter.tensor2numpy_cv2(img_tensor_batch[0].detach())
    plot = plot_boxes_cv2(img, np.array(preds), cls, savename=savename)
    return plot


if __name__ == '__main__':
    target = 'Train'
    w_bbox = 'wo_bbox'
    class_file = 'configs/namefiles/coco.names'
    img_dir = f'./data/INRIAPerson/{target}/pos/'
    patch_path = 'results/exp4/combine/v5/v5-combine-scale-1.png'
    cfg = ConfigParser('./configs/v5-cam.yaml')
    # print(model.device, model.detector.model)

    cls = load_class_names(class_file, trim=False)
    imgs =  os.listdir(img_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loader = dataLoader(img_dir, batch_size=1, is_augment=False, input_size=cfg.DETECTOR.INPUT_SIZE, return_img_name=True)
    attacker = UniversalPatchEvaluator(cfg, patch_path, device)
    model = attacker.detectors[0]

    save_dir = f'./data/cam/{w_bbox}/{model.name}/{target}'
    clean_dir = save_dir + '/clean'
    adv_dir = save_dir + '/adv'
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(adv_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    for img_tensor_batch, img_name in tqdm(loader):
        # img_name = img_name[0].replace('.png', '')
        img_tensor_batch = img_tensor_batch.to(device)
        # ----------------clean-----------------
        # plot = draw_detection(attacker, img_tensor_batch, model, save_dir, f'{cur}-clean-detect.png')
        plot = draw_detection(attacker, img_tensor_batch, save_dir)
        if w_bbox != 'w_bbox':
            plot = FormatConverter.tensor2numpy_cv2(img_tensor_batch[0].detach().cpu())
        # cam = GradCAM(model=model.detector, target_layers=target_layers, use_cuda=True)
        draw_cam(model.detector, img_tensor_batch, plot, clean_dir, img_name[0])

        # ----------------adv-----------------
        img_adv_tensor = attacker.uap_apply(img_tensor_batch)
        # plot = draw_detection(attacker, img_adv_tensor, model, save_dir, f'{cur}-adv-detect.png')
        plot = draw_detection(attacker, img_adv_tensor, save_dir)
        if w_bbox != 'w_bbox':
            plot = FormatConverter.tensor2numpy_cv2(img_adv_tensor[0].detach().cpu())
        draw_cam(model.detector, img_adv_tensor, plot, adv_dir, img_name[0])