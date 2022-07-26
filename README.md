# Base Detection Attack 基础检测攻击框架

## Background 背景介绍

## Install 安装

请直接copy我的环境 
``` bash
cp -r /home/huanghao/anaconda3/envs/dassl/ ~/anaconda3/envs/ # 您需要预先安装anaconda
```

```bash
conda create -n attack -f requirements.txt
conda activate attack
```

---

## Core API 核心API

Base Detection Attack的核心是抽离了检测部门必须提供的API，从而提供一个跟检测无关的对抗攻击框架。也就是说，如果您有一个检测器，您不需要对这个检测器的代码做任何修改，只需要在检测器外层封装一个api.py，Base Detection Attack就可以对您的检测器进行各种攻击测试。


### 攻击框架API

```python
class DetctorAttacker(object):
    # 初始化检测器攻击器 配置文件见./configs/default.yaml
    def __init__(self, cfg):
        pass
    
    # 输入cv格式的图片，检测结果，图片保存路径，可视化检测的结果
    def plot_boxes(self, img, boxes, savename=None):
        pass
    
    # 根据每个预测框获取patch的位置，其中占检测框面积比例的scale参数可以在配置文件中修改
    def get_patch_pos(self, preds, img_cv2):
        pass
    
    # 初始化patches
    def init_patches(self):
        pass

    # 将patch加到图片对应位置
    def apply_patches(self, img_tensor, detector, is_normalize=True):
        pass
```

**具体调用方式请参考attackAPI.py中main函数中的示例。**

---

### 检测部分API(您需要提供的)

api（您需要提供的, 以Faster R-CNN为例子）

``` python
class FRCNN(object):

    # 请务必复制__init__方法的所有内容, detector的类型会被赋值为torch.nn.module
    def __init__(self, name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.device = device
        self.detector = None
        self.name = name

    # 模型梯度清零
    def zero_grad(self):
        self.detector.zero_grad()

    # 载入模型和类别信息，不需要返回值，此函数调用后请保证self.detector()可以检测图片（预处理后的），因不同检测器load参数代码不同，本函数不提供示例，如有需求，请参考./detector_lab/HHDet/military_faster_rcnn/api.py中对应函数编写
    def load(self, detector_weights, classes_path):
        pass

    # 图片准备，输入可能是图片路径，可能是cv2格式的图片，调用此函数进行数据预处理，输出为单张图片处理好的tensor和cv2格式的图片矩阵，例子如下
    def prepare_img(self, img_path=None, img_cv2=None, input_shape=(416, 416)):
        if img_path:
            image = Image.open(img_path)
        else:
            image = Image.fromarray(cv2.cvtColor(img_cv2.astype('uint8'), cv2.COLOR_BGRA2RGBA))
        size = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, [input_shape[1], input_shape[0]])
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        img_tensor = torch.from_numpy(image_data).to(self.device)
        return img_tensor, np.array(image)
    
    # 提供从为预处理的cv2格式的图片到处理后的图片的转换，返回为预处理后的tensor和cv2格式的图片矩阵
    def normalize(self, image_data):
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        img_tensor = torch.from_numpy(image_data).to(self.device)
        img_tensor.requires_grad = True
        return img_tensor, np.array(image_data)
    
    # 提供预处理后好的tensor，返回值为预处理前的图片(cv2格式)和预处理前的图片（cv2格式，整形0-255，可直接保存）
    def unnormalize(self, img_tensor):
        img_numpy = img_tensor.squeeze(0).cpu().detach().numpy() * 255.
        img_numpy = np.transpose(img_numpy, (1, 2, 0))
        img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
        img_numpy_int8 = img_numpy.astype('uint8')
        return img_numpy, img_numpy_int8

    # 检测接口，不直接被外部调用
    def detect(self, img_tensor):
        ...
        return roi_cls_locs, roi_scores, rois, rpn_scores

    # 检测和置信度返回接口，输入预处理后的tensor和原始图片(cv2格式)，返回检测结果[[x1, y1, x2, y2, conf, class_id], ...]和物体置信度（obj_conf），
    def detect_img_tensor_get_bbox_conf(self, input_img, ori_img_cv2):
        ...
        return results, rpn_scores

```

---

## Contact Us 与我们联系

如果您有任何问题，请联系邮箱: huanghao@stu.pku.edu.cn；