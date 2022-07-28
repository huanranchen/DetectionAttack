#!/bin/bash

device=$1
config=$2
patch_name=$config
save=$3
#
#
train_cmd="CUDA_VISIBLE_DEVICES=${device} python train.py \
-cfg=${config}.yaml \
-s=./results/inria/${save}"

echo $train_cmd
eval $train_cmd

# 测试模型自身+迁移的效果
for i in {0..7}
do
  config=inria$i
#  echo $config
  cmd="CUDA_VISIBLE_DEVICES=${device} python evaluate.py -i \
  -p ./results/inris/${save}/patch/1000_${patch_name}.png \
  -cfg ./configs/${config}.yaml \
  -lp /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/labels \
  -dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/pos \
  -s /home/chenziyan/work/BaseDetectionAttack/data/inria/${save} \
  -e 0 &"
  echo cmd
  eval $cmd
done
