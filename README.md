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
# make sure you have these pre-trained detector weights & INRIAPerson labels prepared
├── data
    ├── INRIAPerson
        ├── Train
        ├── Test
            ├── labels
                ├── faster_rcnn-rescale-labels
                ├── ground-truth-rescale-labels
                ├── ssd-rescale-labels
                ├── yolov2-rescale-labels
                ├── yolov3-rescale-labels
                ├── yolov3-tiny-rescale-labels
                ├── yolov4-rescale-labels
                ├── yolov4-tiny-rescale-labels
                └── yolov5-rescale-labels
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
Labels can be downloaded from:
* **BaiduCloud** https://pan.baidu.com/s/1vtkhu2heoXlHNeCEtF2CBA?pwd=o7x8
  * source: detected from the corresponding detectors
  * why rescaled? rescale to [0, 416] to compute AP

Weights can be downloaded from:
* **BaiduCloud** https://pan.baidu.com/s/1tJh-E_0KepziQjsNa8KIJA?pwd=dm85

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

universal-attacker-api（您需要提供的, 以Yolo-v2为例子）

``` python
class HHYolov2(DetectorBase):
    def __init__(self,
                 name, cfg,
                 input_tensor_size=412,
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__(name, cfg, input_tensor_size, device)

    def load(self, model_weights, detector_config_file=None):
        self.detector = Darknet(detector_config_file).to(self.device)
        self.detector.load_weights(model_weights)
        self.eval()

    def detect_test(self, batch_tensor):
        detections_with_grad = self.detector(batch_tensor)
        return detections_with_grad

    def __call__(self, batch_tensor, **kwargs):
        detections_with_grad = self.detector(batch_tensor)  # torch.tensor([1, num, classes_num+4+1])
        # x1, y1, x2, y2, det_conf, cls_max_conf, cls_max_id
        all_boxes, obj_confs, cls_max_ids = get_region_boxes(detections_with_grad, self.conf_thres,
                                            self.detector.num_classes, self.detector.anchors,
                                            self.detector.num_anchors)
        # print(all_boxes[0])
        all_boxes = inter_nms(all_boxes, conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        # print(all_boxes[0])
        obj_confs = obj_confs.view(batch_tensor.size(0), -1)
        cls_max_ids = cls_max_ids.view(batch_tensor.size(0), -1)
        bbox_array = []
        for boxes in all_boxes:
            # boxes = torch.cuda.FloatTensor(boxes)
            # pad_size = self.max_n_labels - len(boxes)
            # boxes = F.pad(boxes, (0, 0, 0, pad_size), value=0).unsqueeze(0)
            if len(boxes):
                boxes[:, :4] = torch.clamp(boxes[:, :4], min=0., max=1.)
            # print(boxes.shape)
            bbox_array.append(boxes)
            # bbox_array = torch.vstack((bbox_array, boxes)) if bbox_array is not None else boxes
        # print(bbox_array)

        output = {'bbox_array': bbox_array, 'obj_confs': obj_confs, "cls_max_ids": cls_max_ids}
        return output

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
#### TODO:通用跨模型攻击

### examples
attack examples from YOLOV3

![image](./results/v3_original.png)

![image](./results/v3_attacked.png)

see more attacked examples in results/


---

## Contact Us 与我们联系

如果您有任何问题，请联系邮箱: huanghao@stu.pku.edu.cn；