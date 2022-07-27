#!/bin/bash

device=$1
config=$2
patch_name=$config
my_date=$(date '+%m-%d')
#
#
echo "CUDA_VISIBLE_DEVICES=${device} python train.py \
-cfg=${config}.yaml \
-s=./results/coco/conf/${my_date}"


# 测试模型自身+迁移的效果
for i in {0..7}
do
  config=inria$i
  echo $config
  echo "CUDA_VISIBLE_DEVICES=${device} python evaluate.py -i \
  -p ./results/inria/conf/${my_date}/patch/1000_${patch_name}.png \
  -cfg ./configs/${config}.yaml \
  -lp /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/labels \
  -dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Test/pos \
  -s /home/chenziyan/work/BaseDetectionAttack/data/inria/conf/${my_date} \
  -e 0"
done
