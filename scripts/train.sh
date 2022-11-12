
cuda=$1
target=$2
model=$3
name=$4
args=$6

if [ ! -n "$5" ] ;then
  folder=exp4
else
  folder=$5
fi

mkdir ./results/$folder
mkdir ./results/$folder/$target
mkdir ./results/$folder/$target/$model

CUDA_VISIBLE_DEVICES=$cuda nohup python train_optim.py \
-cfg=$target/$model.yaml \
-s=./results/$folder/$target/$model/ \
-n=$model-$name \
$args \
>./results/$folder/$target/$model/$model-$name.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python train_optim.py -cfg=method/v5-sgd-p9.yaml -s ./results/exp6/method/v5 -n v5-sgd-p9-1 >./results/