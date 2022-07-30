#!/bin/bash

screen -S military
cd ~/work/BaseDetectionAttack
conda activate dassl

CUDA_VISIBLE_DEVICES=5 python train.py -cfg=ensemble.yaml -s=./results/inria/ensemble/$(date '+%m-%d')
nohup bash train.sh 4 inria7 >./results/inria/$(date '+%m-%d')/inria7.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python train.py -cfg=inria2.yaml -s=./results/inria/$(date '+%m-%d')


CUDA_VISIBLE_DEVICES=2 python evaluate.py -i -g \
-p ./results/hr/patch_1000.pth \
-cfg ./configs/inria7.yaml \
-lp /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/pos \
-s /home/chenziyan/work/BaseDetectionAttack/data/inria/test/$(date '+%m-%d') \
-e 0 \
-d YOLOV3 YOLOV3-TINY YOLOV4 YOLOV4-TINY FASTER-RCNN

CUDA_VISIBLE_DEVICES=2 python evaluate.py -i \
-p ./results/naco/patch_1000.pth \
-cfg ./configs/coco80.yaml \
-lp /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/pos \
-s /home/chenziyan/work/BaseDetectionAttack/data/inria/$(date '+%m-%d') \
-e 0

CUDA_VISIBLE_DEVICES=2 python evaluate.py -i -l \
-p ./results/military/patch/595parallel.png \
-cfg ./configs/parallel.yaml

CUDA_VISIBLE_DEVICES=3 python entry.py --attack_method=serial --cfg=inria3.yaml --cuda=3

####################For coco-patch test in INRIA
CUDA_VISIBLE_DEVICES=3 python evaluate.py \
-p ./results/coco/conf/07-22/patch/400_0_inria0.png \
-cfg ./configs/inria0.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/pos \
-s /home/chenziyan/work/BaseDetectionAttack/data/inria/Test/conf/$(date '+%m-%d') \
-e 0

####################For test in coco
CUDA_VISIBLE_DEVICES=3 python evaluate.py -i \
-p ./results/coco/conf/07-22/patch/400_0_inria0.png \
-cfg ./configs/inria0.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/train/train2017_labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/train/train2017 \
-s /home/chenziyan/work/BaseDetectionAttack/data/coco/conf/$(date '+%m-%d') \
-e -1 \
> /home/chenziyan/work/BaseDetectionAttack/data/coco/partial/$(date '+%m-%d')-eva.log


CUDA_VISIBLE_DEVICES=3 python evaluate.py -i -o \
-p ./results/coco/conf/07-22/patch/400_0_inria0.png \
-cfg ./configs/inria0.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/val/val2017_labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/val/val2017 \
-s /home/chenziyan/work/BaseDetectionAttack/data/coco/$(date '+%m-%d') \
-e -1

CUDA_VISIBLE_DEVICES=0 python test.py --config_file=./configs/parallel.yaml --patch=./test/patch.png --save=./test