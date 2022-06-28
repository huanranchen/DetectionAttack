#!/bin/bash

screen -S military
cd ~/work/BaseDetectionAttack
conda activate dassl

CUDA_VISIBLE_DEVICES=2 python evaluate.py --config_file=./configs/parallel.yaml --patch=./test/patch.png --save=./test

CUDA_VISIBLE_DEVICES=0 python militaryAttack.py --attack_method=parallel --config_file=parallel.yaml

# partial
CUDA_VISIBLE_DEVICES=3 python attackAPI.py -p --attack_method=parallel \
--cfg=coco0.yaml \
-s ./results/coco/$(date '+%m-%d') \
> ./results/coco/$(date '+%m-%d')-train.log

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

####################For coco-patch test in INRIA
CUDA_VISIBLE_DEVICES=0 python evaluate.py -i -o \
-p ./results/coco/5-26/patch/5_111000_coco0.png \
-cfg ./configs/coco0.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/pos \
-s /home/chenziyan/work/BaseDetectionAttack/data/inria/$(date '+%m-%d') \
-e [0]

####################For test in coco
CUDA_VISIBLE_DEVICES=3 python evaluate.py -i -o \
-p ./results/coco/partial/patch/1_17000_coco2.png \
-cfg ./configs/coco2.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/test/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/test/test2017 \
-s /home/chenziyan/work/BaseDetectionAttack/data/coco/partial/$(date '+%m-%d') \
-e 1 \
> /home/chenziyan/work/BaseDetectionAttack/data/coco/partial/$(date '+%m-%d')-eva.log


CUDA_VISIBLE_DEVICES=3 python evaluate.py -i -o \
-p ./results/coco/5-26/patch/5_111000_coco0.png \
-cfg ./configs/coco0.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/test/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/test/test2017 \
-s /home/chenziyan/work/BaseDetectionAttack/data/coco/$(date '+%m-%d') \
-e -1

CUDA_VISIBLE_DEVICES=0 python test.py --config_file=./configs/parallel.yaml --patch=./test/patch.png --save=./test