#!/bin/bash
# nohup bash train.sh 1 ps4 $(date '+%m-%d') >./results/inria/07-28/ps4.log 2>&1 &
gg="-g"
device=$1
patch_name=$2
save=$3
A=$4
target=$5
ifng=$6

if  [ ! -n "$ifng" && $ifng = "-ng" ] ;then
    gg=""
fi

# 测试模型自身+迁移的效果
for i in ${A[@]}
do
  config=$i
  echo $config
  echo "./results/${patch_name}.png"
#  echo $config inria/${save}/patch/1000_
  cmd="CUDA_VISIBLE_DEVICES=${device} python evaluate.py -i $gg\
  -p ./results/${patch_name}.png \
  -cfg ./configs/${i}.yaml \
  -lp /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/$target/labels \
  -dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/$target/pos \
  -s /home/chenziyan/work/BaseDetectionAttack/data/inria/${save} \
  -e 0 &"
  echo cmd
  eval $cmd
  eval "sleep 10"
done
