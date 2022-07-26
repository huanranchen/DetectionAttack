import sys
import cv2

from detector_lab.HHDet.yolov5.api_backup import HHYolov5

img_path = '/home/chenziyan/work/BaseDetectionAttack/data/coco/train/train2017/000000000030.jpg'
sys.path.append('~/work/BaseDetectionAttack/detector_lab/HHDet/yolov5')
hhyolov5 = HHYolov5(name="YOLOV5")
hhyolov5.load(model_weights='./detector_lab/HHDet/yolov5/yolov5/weight/yolov5s.pt')
hhyolov5.detect_cv2_show(img_path)

im0s = cv2.imread(img_path)
# Padded resize
im = letterbox(im0s, hhyolov5.imgsz, stride=hhyolov5.stride, auto=True)[0]
# Convert
im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
im = np.ascontiguousarray(im)

im = torch.from_numpy(im).to(hhyolov5.device).float()
im /= 255  # 0 - 255 to 0.0 - 1.0
if len(im.shape) == 3:
    im = im[None]  # expand for batch dim

img_tensor = Variable(im, requires_grad=True)
box_array, confs = hhyolov5.detect_img_tensor_get_bbox_conf(img_tensor)
loss = hhyolov5.temp_loss(confs)
loss.backward()
print(img_tensor.grad, img_tensor.grad.size())