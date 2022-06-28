#!/bin/bash

# partial: attack unseen class
#CUDA_VISIBLE_DEVICES=$1 python evaluate.py -i -o \
#-p ./results/coco/partial/patch/2_99000_coco2.png \
#-cfg ./configs/coco2.yaml \
#-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/test/labels \
#-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/test/test2017 \
#-s /home/chenziyan/work/BaseDetectionAttack/data/coco/partial/$(date '+%m-%d') \
#-e -3

#test_path="9_70000_coco0"
CUDA_VISIBLE_DEVICES=0 python evaluate.py -i -o \
-p ./results/coco/5-26/patch/0_1000_coco0.png \
-cfg ./configs/coco0.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/test/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/test/test2017 \
-s /home/chenziyan/work/BaseDetectionAttack/data/coco/06-01 \
-e -1

# 普通全类别训练在coco上作验证：全类别验证
CUDA_VISIBLE_DEVICES=0 python evaluate.py -i -o \
-p ./results/coco/5-26/patch/1_99000_coco0.png \
-cfg ./configs/coco0.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/test/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/test/test2017 \
-s /home/chenziyan/work/BaseDetectionAttack/data/coco/06-01 \
-e -1

CUDA_VISIBLE_DEVICES=0 python evaluate.py -i -o \
-p ./results/coco/5-26/patch/4_99000_coco0.png \
-cfg ./configs/coco0.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/test/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/test/test2017 \
-s /home/chenziyan/work/BaseDetectionAttack/data/coco/06-01 \
-e -1

CUDA_VISIBLE_DEVICES=0 python evaluate.py -i -o \
-p ./results/coco/5-26/patch/7_99000_coco0.png \
-cfg ./configs/coco0.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/test/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/test/test2017 \
-s /home/chenziyan/work/BaseDetectionAttack/data/coco/06-01 \
-e -1

# 普通全类别训练在INRIA上的迁移性
#CUDA_VISIBLE_DEVICES=$1 python evaluate.py -i -o \
#-p ./results/coco/5-26/patch/"${test_path}".png \
#-cfg ./configs/coco0.yaml \
#-gt /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/labels \
#-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/pos \
#-s /home/chenziyan/work/BaseDetectionAttack/data/inria/$(date '+%m-%d') \
#-e 0
