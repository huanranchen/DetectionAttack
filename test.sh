#!/bin/bash
# nohup bash train.sh 1 ps4 $(date '+%m-%d') >./results/inria/07-28/ps4.log 2>&1 &
gg="-g"
ii="-i"
device=$1
patch_name=$2
save=$3
A=$4
targets=$5
ifng=$6
ifim=$7
echo 'ifng'$ifng
if [ $ifng = "-ng" ] ;then
    gg=""
fi
if [ $ifng = "-ni" ] ;then
    ii=""
fi

# 测试模型自身+迁移的效果
for config in ${A[@]}
do
  echo $config
  echo "./results/${patch_name}.png"
  for target in ${targets[@]}
  do
    cmd="CUDA_VISIBLE_DEVICES=${device} nohup python evaluate.py $ii $gg\
    -p ./results/${patch_name}.png \
    -cfg ./configs/${config}.yaml \
    -lp /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/$target/labels \
    -dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/$target/pos \
    -s /home/chenziyan/work/BaseDetectionAttack/data/${save}/${target} \
    -e 0 &"
    echo $cmd
    eval $cmd
    sleep 2
  done
done
