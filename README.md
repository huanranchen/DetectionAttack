# Base Detection Attack 基础检测攻击框架

## Background 背景介绍

## Install 安装
### Environment
```bash
conda create -n dassl python=3.7
conda activate dassl
pip install -r requirements.txt
```

### Models
```bash
# make sure you have these pre-trained detector weight files prepared
└── detlib
    ├── base.py
    ├── HHDet
    ├── torchDet
    └── weights
        ├── setup.sh
        ├── yolov2.weights
        ├── yolov3-tiny.weights
        ├── yolov3.weights
        ├── yolov4.pth
        ├── yolov4-tiny.weights
        ├── yolov4.weights
        ├── yolov5n.pt
        ├── yolov5s6.pt
        └── yolov5s.pt
bash ./detlib/weights/setup.sh
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
    def apply_patches(self, img_tensor, detector, attacking=False):
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

### implementation
配置参数在配置文件中进行设置

部分配置参数仍定义在文件内部，例如是否保存plot、图像文件夹路径等
```shell
# for attack: config as configs/inria.yaml
python attackAPI.py

# for military attack: config as configs/default.yaml
python militaryAttackAPI.py
```

### utils
增加的工具类函数


```python
# data_reader.py
# 拟data loader图片准备：读入names中的img，并输出为 numpy batch
def read_img_np_batch(names, input_size):
    # return format: RGB [b, 3, h, w] (dtype=uint8)
    pass
    return img_numpy_batch

# utils.py
# 检测器间NMS
def inter_nms():
    # input: [N*6] preds list, list len=batch size
    pass
```

### 攻击框架API
#### 增加的通用跨模型部分
```python
# UniversalDetectorAttacker为继承类
class UniversalDetectorAttacker(DetctorAttacker):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
    
    # 初始化通用的攻击Patch
    def init_universal_patch(self, height=100, width=100):
        # save to self.universal_patch: [1, 3, height, width] tensor
        universal_patch = np.random.randint(low=0, high=255, size=(height, width, 3))
        universal_patch = np.expand_dims(np.transpose(universal_patch, (2, 0, 1)), 0)
        self.universal_patch = torch.from_numpy(np.array(universal_patch, dtype='float32')/255.).to(self.device)
        self.universal_patch.requires_grad = True
    
    # 根据攻击类别过滤预测框，并规范化检测框格式
    def get_patch_pos_batch(self, all_preds):
        # input all_preds: [N*6]的检测框list，其长度等于batch大小，bbox坐标为0~1
        # results save to self.batch_boxes
        pass
    
    # 根据已获取的规范化检测框，在图像中加入通用的攻击patch
    def uap_apply(self, numpy_batch, attacking=False):
        # 同时对universal_patch进行该检测器规定的预处理 detector.normalize_tensor()
        pass
    
    # 合并多个检测器的预测结果
    def merge_batch_pred(self, all_preds, preds):
        pass
        # return: 返回所有batch中是否存在一个以上有效目标
        return has_target

    # 所有检测器对 攻击后的img_batch 的检测结果，将batch的第一张保存到results/下
    def adv_detect_save(self, img_numpy_batch, save_postfix):
        pass
```


### 检测框架API
#### 通用跨模型增加部分
以FRCNN为例
```python
class FRCNN(object):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
    
    # numpy img batch转化为tensor batch，并进行该检测器规定的预处理
    def init_img_batch(self, img_numpy_batch):
        # 1、init numpy(RGB) batch (b, c, h, w) into (deviced) tensor batch
        # 2、preprocessing by internal detector e.g. mean-std normalization
        pass
    
    # 对img batch进行目标检测，返回检测框及置信度
    def forward(self, batch_tensor):
        # 功能为batch版本的detect_img_tensor_get_bbox_conf()
        pass
        return preds, confs
    
    # 对tensor进行检测器规定的normalize操作
    def normalize_tensor(self, tensor_data):
        # 如果需要normalize，需要对tensor_data的clone()进行操作
        # 否则直接返回即可
        return tensor_data
    
    # 没有特殊添加功能，仅为复用所改写函数：不直接被外部调用
    def box_rectify(self, result, image_shape):
        pass
```

### Pipeline elaboration
#### 通用跨模型攻击
一个带删减的简略pipeline

```python
#初始化clean的图像batch: RGB [b, 3, h, w] numpy
read_img_np_batch()
```


```python
# 调用所有检测器对 img_numpy_batch 进行目标检测
for detector in detectors:
    detector()
# 检测器间NMS
inter_nms()
# 获取所有的target初始目标locations
has_target = get_patch_pos_batch()

# 检查当前batch图像是否包含一个以上检测框，否则不进行攻击
if not has_target:
    continue
```


```python
# Sequential or Parallel attack
for i in range(cfg.MAX_ITER):
    for detector in detectors:
        # 获取攻击后的img_tensor
        adv_img_tensor = uap_apply()
        # 进行迭代攻击，（串/并行）更新universal_patch
        non_targeted_attack_batch(adv_img_tensor, detector)
```
目前攻击只写了第一种串行更新的方法
在条件
MAX_ITER=100，
ITER_STEP=2，
BATCH_SIZE=2下，
攻击一整轮所需时间:

对于military YOLOX、FAST RCNN：80s左右;

对于YOLOV4、YOLOV3：50s左右

### examples
attack examples from YOLOV3

![image](./results/v3_original.png)

![image](./results/v3_attacked.png)

see more attacked examples in results/


---

## Contact Us 与我们联系

如果您有任何问题，请联系邮箱: huanghao@stu.pku.edu.cn；