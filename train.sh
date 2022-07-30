#!/bin/bash
# nohup bash train.sh 1 ps4 $(date '+%m-%d') >./results/inria/07-28/ps4.log 2>&1 &
device=$1
config=$2
patch_name=$config
save=$3


train_cmd="CUDA_VISIBLE_DEVICES=${device} python train.py \
-cfg=${config}.yaml \
-s=./results/inria/${save}"

echo $train_cmd
eval $train_cmd
sleep 2

A=$4
targets=("Train" "Test")
# 测试模型自身+迁移的效果
for config in ${A[@]}
do
  echo $config
  for target in ${targets[@]}
  do
    cmd="CUDA_VISIBLE_DEVICES=${device} python evaluate.py -i -g\
    -p ./results/inria/${save}/patch/1000_${patch_name}.png \
    -cfg ./configs/${config}.yaml \
    -lp /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/$target/labels \
    -dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/$target/pos \
    -s /home/chenziyan/work/BaseDetectionAttack/data/inria/${save}/$target \
    -e 0 &"
    echo $cmd
    eval $cmd
  done
  sleep 2
done
