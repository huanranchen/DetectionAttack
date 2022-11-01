#!/bin/bash
# nohup bash train.sh 1 ps4 $(date '+%m-%d') >./results/inria/07-28/ps4.log 2>&1 &
gg="-g"
device=$1
patch_name=$2
save=$3
A=$4
targets=$5
ifng=$6
ifim=$7
echo 'ifng: '$ifng
if [ $ifng = "-ng" ] ;then
    gg=""
fi

# CCTVfootage
lp="CCTVfootage/labels"
dr="CCTVfootage/imgs"
save_dir="ccvtv_footage"

# COCO person val
lp="COCOperson/val/val2017/labels"
dr="COCOperson/val/val2017/pos"
save_dir="coco_person"

for config in ${A[@]}
do
  echo $config
  echo "./results/${patch_name}.png"
    cmd="CUDA_VISIBLE_DEVICES=${device} python evaluate.py $ifim $gg\
    -p ./results/${patch_name}.png \
    -cfg ./configs/${config}.yaml \
    -lp ~/work/BaseDetectionAttack/data/$lp \
    -dr ~/work/BaseDetectionAttack/data/$dr \
    -s ~/work/BaseDetectionAttack/data/$save_dir/${save}/ \
    -e 0 &"
    echo $cmd
    eval $cmd
    sleep 2
done
