import time

import cv2
import torch
import torchvision
import numpy as np
import math


def toTensor(img):
    assert type(img) == np.ndarray,'the img type is {}, but ndarry expected'.format(type(img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().cuda().unsqueeze(0)  


class LoadImage(object):
    """A simple pipeline to load image."""
    def __init__(self, detector_name):
        self.detector_name = detector_name

    def __call__(self, results):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if self.detector_name == "FasterRCNN":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])]) 

        if self.detector_name == "SSD":
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[1, 1, 1])]) 

        img = transform(results['img'].squeeze(0)).unsqueeze(0)
      
        results['img'] = [img]
        return results


def order_points(pts):
    #将得到的矩形顶点顺时针排序
    center = pts.sum(axis=0) / 4
    deltaxy = pts - np.tile(center, (4, 1))
    rad = np.arctan2(deltaxy[:, 1], deltaxy[:, 0])
    sortidx = np.argsort(rad)
    return pts[sortidx]


def four_point_transform(src, w, h):
    #计算变换
    src = src.astype(np.float32)
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src, dst)
    return M


def initialize(black_img, white_img, resize_w, resize_h, gray_thresh = 70):
    """
    black_img:屏幕为黑色时的图片绝对路径
    white_img:屏幕为白色时的图片绝对路径
    resize_w:变换之后的宽
    resize_h:变换之后的高
    gray_thresh:二值化时的灰度阈值，影响提取效果
    返回值:M为变换，mask为屏幕掩膜
    """

    black = cv2.resize(cv2.imread(black_img), (resize_w, resize_h))
    white = cv2.resize(cv2.imread(white_img), (black.shape[1], black.shape[0]))
    roi = cv2.subtract(white,black)
    
    #处理图像
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)
    thresh = cv2.threshold(sharpen,gray_thresh,255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    
    #寻找轮廓
    h = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours = h[0]
    
    area = [cv2.contourArea(c) for c in contours]
    max_idx = np.argmax(area)
    #排除掉外边框
    if(area[max_idx] > (black.shape[0]*black.shape[1]*2/3)):
        max_idx = np.argsort(-np.array(area))[1]
    
    approx = cv2.approxPolyDP(contours[max_idx],10,True)
    
    assert len(approx)==4,"提取不成功，请重新设置图像二值化阈值"
    
    #生成mask
    mask = np.zeros(white.shape,np.uint8)
    cv2.fillPoly(mask, [approx], (255, 255, 255))

    cv2.imwrite('./mask.png', mask)
    
    #生成变换
    ordered_pts = order_points(approx.reshape(-1,2))

    ordered_pts = np.array([ordered_pts[2], ordered_pts[1], ordered_pts[0], ordered_pts[3]])


    M = four_point_transform(ordered_pts, resize_w, resize_h)
    
    #返回变换
    return M, mask


def transform(image, M, mask, resize_w, resize_h):
    """
    image_path:需要变换的图片
    M:initialize函数返回的变换
    mask:initialize函数返回的掩膜
    resize_w:和initialize函数保持一致
    resize_h:和initialize函数保持一致
    返回值:变换后的图片（cv2）
    """
    # image = cv2.imread(image_path)

    # image = cv2.resize(cv2.imread(image_path), (800, 800))
    image = np.uint8(image)
    # mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    temp = image & mask
    cv2.imwrite('./mask_p.png', temp)
    warped = cv2.warpPerspective(temp, M, (resize_w, resize_h))
    # warped = cv2.warpPerspective(temp, M, (resize_h, resize_w))

    return warped


class obj(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, obj(b) if isinstance(b, dict) else b)



