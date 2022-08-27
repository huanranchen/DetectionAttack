#!/bin/bash
# nohup bash train.sh 1 ps4 $(date '+%m-%d') >./results/inria/07-28/ps4.log 2>&1 &
device=$1
config=$2
file_name=$3
save=$4
log_name=$5

train_cmd="CUDA_VISIBLE_DEVICES=${device} nohup python ${file_name} \
-cfg=${config}.yaml \
-s=${save} >${save}/${log_name} 2>&1 &"

echo $train_cmd
eval $train_cmd
#sleep 2

#A=$4
#targets=("Test")
## 测试模型自身+迁移的效果
#for config in ${A[@]}
#do
#  echo $config
#  eval "bash test.sh $device ${save}/patch/${patch_name} ${save} (coco80 coco91) $targets"
##  for target in ${targets[@]}
##  do
##    cmd="CUDA_VISIBLE_DEVICES=${device} python evaluate.py -i -g\
##    -p ./results/${save}/patch/${patch_name}.png \
##    -cfg ./configs/${config}.yaml \
##    -lp /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/$target/labels \
##    -dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/$target/pos \
##    -s /home/chenziyan/work/BaseDetectionAttack/data/${save}/$target \
##    -e 0 &"
##    echo $cmd
##    eval $cmd
##  done
##  sleep 2
#done
