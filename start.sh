#!/bin/bash

screen -S military
cd ~/work/BaseDetectionAttack
conda activate dassl

CUDA_VISIBLE_DEVICES=2 python train.py -cfg=inria7.yaml -s=./results/inria/conf/$(date '+%m-%d')

CUDA_VISIBLE_DEVICES=3 nohup python train.py -cfg=ps3.yaml -s=./results/inria/conf/ps/$(date '+%m-%d') > ./results/ps3.out 2>&1 &

CUDA_VISIBLE_DEVICES=3 python train.py -cfg=sign.yaml -s=./results/coco/sign/$(date '+%m-%d')

CUDA_VISIBLE_DEVICES=2 python evaluate.py --config_file=./configs/parallel.yaml --patch=./test/patch.png --save=./test

CUDA_VISIBLE_DEVICES=1 python militaryAttack.py --attack_method=parallel --config_file=parallel.yaml

# partial
CUDA_VISIBLE_DEVICES=3 python attackAPI.py -p --attack_method=parallel \
--cfg=coco0.yaml \
-s ./results/coco/$(date '+%m-%d') \
> ./results/coco/$(date '+%m-%d')-train.log

CUDA_VISIBLE_DEVICES=2 python evaluate.py -i \
-p ./results/inria/conf/ps/07-25/patch/400_0_ps5.png \
-cfg ./configs/ps5.yaml \
-lp /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/pos \
-s /home/chenziyan/work/BaseDetectionAttack/data/inria/conf/ps/$(date '+%m-%d') \
-e 0 \
-d YOLOV3 YOLOV3-TINY YOLOV4 YOLOV4-TINY FASTER-RCNN

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