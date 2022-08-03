#!/bin/bash
# nohup bash train.sh 1 ps4 $(date '+%m-%d') >./results/inria/07-28/ps4.log 2>&1 &
gg="-g"
device=$1
patch_name=$2
save=$3
A=$4
ifng=$5
echo ifng+$ifng
if [ ! -n $ifng && $ifng = "-ng" ] ;then
    gg=""
fi
echo "gg "+$gg
# 测试模型自身+迁移的效果
for config in ${A[@]}
do
  echo $config
  echo "./results/${patch_name}.png"
  cmd="CUDA_VISIBLE_DEVICES=${device} python evaluate.py $gg\
  -p ./results/${patch_name}.png \
  -cfg ./configs/${config}.yaml \
  -lp /home/chenziyan/work/BaseDetectionAttack/data/coco/val/val2017_labels \
  -dr /home/chenziyan/work/BaseDetectionAttack/data/coco/val/val2017 \
  -s /home/chenziyan/work/BaseDetectionAttack/data/${save}/${target} \
  -e 0 &"
  echo $cmd
  eval $cmd
done
