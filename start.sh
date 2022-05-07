#!/bin/bash

screen -S military
cd ~/work/BaseDetectionAttack
conda activate dassl
CUDA_VISIBLE_DEVICES=2 python militaryAttackAPI.py --postfix=10

CUDA_VISIBLE_DEVICES=0 python evaluate.py

CUDA_VISIBLE_DEVICES=1 python militaryAttackAPI.py --attack_method=serial --config_file=military5.yaml
