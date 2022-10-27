
cuda=$1
target=$2
model=$3
name=$4

if [ ! -n "$5" ] ;then
  folder=exp4
else
  folder=$5
fi

if [ ! -n "$6" ] ;then
  combine=""
else
  combine="-"$6
fi

mkdir ./results/$folder
mkdir ./results/$folder/method
mkdir ./results/$folder/method/$model

save_dir=./results/$folder/method/$model/
save_name=$model-$name$combine

CUDA_VISIBLE_DEVICES=$cuda nohup python train_pgd.py \
-cfg=method/$model-$target$combine.yaml \
-s=$save_dir \
-n=$save_name \
>$save_dir/$save_name.log 2>&1 &