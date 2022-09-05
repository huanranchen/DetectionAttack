#Evaluation of universal patches

### mAP.py
Acknowledgement to [mAP computation source](https://github.com/Cartucho/mAP).

## Generate labels of detection
make sure your dir 'labels' like the these.
```bash
# dir-name rule: detector_name-labels
├── labels
│   ├── ground-truth
│   ├── faster_rcnn-labels
│   ├── ssd-labels
│   ├── yolov3-labels
│   ├── yolov3-tiny-labels
│   ├── yolov4-labels
│   ├── yolov4-tiny-labels
│   └── yolov5-labels
```
The 'ground-truth' is the annotation labels, while the others are from detections of given detectors.
This is for different standards to compute mAP.

### Ground-truth labels
We have the **coco_process.py**、**inria_process.py** to help process annotation labels.

```bash
python ./data/coco_process.py \
--img_folder=data/coco/val/val2017 \
--name_file=../configs/namefiles/coco-91.names \
--json_path=./coco/instances_val2017.json \  
--save_path=./coco/val/val2017_labels/ground-truth
```

### xx_process.txt
This is to follow the format needed by mAP.py.

xx_precess.py is for processing annotations of the dataset into a standard yolo-style label format as:

```
cls_name x_min y_min x_max y_max

person 0.1345 0.4567 0.2456 0.9876
```

where cls_number is the index of the label in the class name file, and the xyxy of the bbox is scale into [0, 1].


### Detection labels
We have the **gen_det_labels.py** to help generate detection labels with help with our Detection-Attack framework.
```bash
python ./data/gen_det_labels.py \
-dr=data/coco/train/train2017 \
-sr=data/coco/train/train2017_labels \
-cfg=coco80.yaml

python ./data/gen_det_labels.py \
-dr=data/coco/train/train2017 \
-sr=data/coco/train/train2017_labels \
-cfg=coco91.yaml
```