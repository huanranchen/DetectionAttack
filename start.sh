#!/bin/bash

screen -S military
cd ~/work/BaseDetectionAttack
conda activate dassl

CUDA_VISIBLE_DEVICES=0 python evaluate.py --config_file=./configs/parallel.yaml --patch=./test/patch.png --save=./test

CUDA_VISIBLE_DEVICES=0 python militaryAttack.py --attack_method=parallel --config_file=parallel.yaml

CUDA_VISIBLE_DEVICES=1 python attackAPI.py -p --attack_method=parallel --cfg=coco1.yaml -s ./results/coco

CUDA_VISIBLE_DEVICES=0 python evaluate.py -i \
-p ./results/inria/patch/99_inria3.png \
-cfg ./configs/inria3.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/pos \
-s /home/chenziyan/work/BaseDetectionAttack/data/inria

CUDA_VISIBLE_DEVICES=2 python evaluate.py -i -l \
-p ./results/military/patch/595parallel.png \
-cfg ./configs/parallel.yaml

CUDA_VISIBLE_DEVICES=3 python entry.py --attack_method=serial --cfg=inria3.yaml --cuda=3

####################For test in INRIA
CUDA_VISIBLE_DEVICES=3 python evaluate.py -i -o \
-p ./results/coco/patch/2_42000_coco1.png \
-cfg ./configs/coco1.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/pos \
-s /home/chenziyan/work/BaseDetectionAttack/data/coco/5-25

####################For test in coco
CUDA_VISIBLE_DEVICES=3 python evaluate.py -i -o \
-p ./results/coco/patch/2_42000_coco1.png \
-cfg ./configs/coco1.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/test/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/test/test2017 \
-s /home/chenziyan/work/BaseDetectionAttack/data/coco/5-25


CUDA_VISIBLE_DEVICES=0 python test.py --config_file=./configs/parallel.yaml --patch=./test/patch.png --save=./test