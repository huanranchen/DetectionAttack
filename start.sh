#!/bin/bash

screen -S military
cd ~/work/BaseDetectionAttack
conda activate dassl

CUDA_VISIBLE_DEVICES=2 python evaluate.py --config_file=parallel.yaml -p ./results/military/patch/50parallel.png

CUDA_VISIBLE_DEVICES=1 python militaryAttackAPI.py --attack_method=serial --config_file=military5.yaml

CUDA_VISIBLE_DEVICES=0 python attackAPI.py --attack_method=serial --cfg=inria1.yaml
