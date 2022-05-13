#!/bin/bash

screen -S military
cd ~/work/BaseDetectionAttack
conda activate dassl

CUDA_VISIBLE_DEVICES=2 python evaluate.py --config_file=parallel.yaml -p ./results/military/patch/50parallel.png

CUDA_VISIBLE_DEVICES=2 python militaryAttack.py --attack_method=parallel --config_file=parallel.yaml

CUDA_VISIBLE_DEVICES=2 python attackAPI.py --attack_method=serial --cfg=inria0.yaml -s ./results/YOLOV4

CUDA_VISIBLE_DEVICES=2 python evaluate.py -o \
-p ./results/inria/patch/99_inria1.png \
-cfg ./configs/inria1.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/pos \
-s /home/chenziyan/work/BaseDetectionAttack/data/inria

CUDA_VISIBLE_DEVICES=2 python evaluate.py -i -l \
-p ./results/military/patch/595parallel.png \
-cfg ./configs/parallel.yaml

CUDA_VISIBLE_DEVICES=3 python entry.py --attack_method=serial --cfg=inria3.yaml --cuda=3
