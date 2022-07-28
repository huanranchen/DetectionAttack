#!/bin/bash

device=$1
val_path=./results/inria/$2
echo "${val_path}"

# partial: attack unseen class
#CUDA_VISIBLE_DEVICES=3 python evaluate.py -i -o \
#-p ./results/coco/partial/patch/8_99000_coco2.png \
#-cfg ./configs/coco2.yaml \
#-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/train/train2017_labels \
#-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/train/train2017 \
#-s /home/chenziyan/work/BaseDetectionAttack/data/coco/partial/$(date '+%m-%d') \
#-e -2

CUDA_VISIBLE_DEVICES=${device} python evaluate.py -i \
-p ${val_path} \
-cfg ./configs/coco80.yaml \
-lp /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/pos \
-s /home/chenziyan/work/BaseDetectionAttack/data/inria/$(date '+%m-%d') \
-e 0 &

# 普通全类别训练在coco上作验证：全类别验证
#CUDA_VISIBLE_DEVICES=3 python evaluate.py -i -o \
#-p ./results/inria/conf/07-13/patch/830_0_inria0.png \
#-cfg ./configs/inria0.yaml \
#-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/val/val2017_labels \
#-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/val/val2017 \
#-s /home/chenziyan/work/BaseDetectionAttack/data/coco/conf/inria/$(date '+%m-%d') \
#-e -1 \
#-d YOLOV3 YOLOV4

# 普通全类别训练在INRIA上的迁移性
CUDA_VISIBLE_DEVICES=${device} python evaluate.py -i \
-p ${val_path} \
-cfg ./configs/coco91.yaml \
-lp /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/pos \
-s /home/chenziyan/work/BaseDetectionAttack/data/inria/$(date '+%m-%d') \
-e 0 &