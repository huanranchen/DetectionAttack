#!/bin/bash

screen -S military
cd ~/work/BaseDetectionAttack
conda activate dassl

CUDA_VISIBLE_DEVICES=2 python evaluate.py --config_file=parallel.yaml -p ./results/military/patch/50parallel.png

CUDA_VISIBLE_DEVICES=1 python militaryAttackAPI.py --attack_method=serial --config_file=military5.yaml

CUDA_VISIBLE_DEVICES=2 python attackAPI.py --attack_method=serial --cfg=inria4.yaml

CUDA_VISIBLE_DEVICES=2 python evaluate.py -o \
-p ./results/inria/patch/99_inria0.png \
-cfg ./configs/inria0.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/pos \
-s /home/chenziyan/work/BaseDetectionAttack/data/inria
