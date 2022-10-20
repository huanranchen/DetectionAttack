
cuda=$1
target=$2
model=$3
name=$4

mkdir ./results/exp4/$target/$model

CUDA_VISIBLE_DEVICES=$cuda nohup python train_optim.py \
-cfg=$target/$model.yaml \
-s=./results/exp4/$target/$model/ \
-n=$model-$name \
>./results/exp4/$target/$model/$model-$name.log 2>&1 &