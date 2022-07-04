#!/bin/bash
test_path=./results/coco/partial/patch/8_99000_coco2.png
echo "${test_path}"
# partial: attack unseen class
CUDA_VISIBLE_DEVICES=3 python evaluate.py -i -o \
-p ./results/coco/partial/patch/8_99000_coco2.png \
-cfg ./configs/coco2.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/test/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/test/test2017 \
-s /home/chenziyan/work/BaseDetectionAttack/data/coco/partial/$(date '+%m-%d') \
-e -2

CUDA_VISIBLE_DEVICES=$1 python evaluate.py -i -o \
-p ./results/coco/partial/patch/8_99000_coco2.png \
-cfg ./configs/coco2.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/pos \
-s /home/chenziyan/work/BaseDetectionAttack/data/inria/$(date '+%m-%d') \
-e 0

# 普通全类别训练在coco上作验证：全类别验证
#CUDA_VISIBLE_DEVICES=0 python evaluate.py -i -o \
#-p ./results/coco/5-26/patch/7_99000_coco0.png \
#-cfg ./configs/coco0.yaml \
#-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/test/labels \
#-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/test/test2017 \
#-s /home/chenziyan/work/BaseDetectionAttack/data/coco/06-01 \
#-e -1

# 普通全类别训练在INRIA上的迁移性
#CUDA_VISIBLE_DEVICES=$1 python evaluate.py -i -o \
#-p ./results/coco/5-26/patch/"${test_path}".png \
#-cfg ./configs/coco0.yaml \
#-gt /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/labels \
#-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/pos \
#-s /home/chenziyan/work/BaseDetectionAttack/data/inria/$(date '+%m-%d') \
#-e [0]
from tools.parser import ConfigParser
cfg = ConfigParser('./configs/coco2.yaml')
eva_class = cfg.rectify_class_list('-2', dtype='str')
eva_class
print('Eva(Attack) classes from evaluation: ', cfg.show_class_index(eva_class))
print('Eva classes names from evaluation: ', eva_class)
ignore_class = list(set(cfg.all_class_names).difference(set(eva_class)))
ignore_class