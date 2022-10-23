
cuda=$1
target=$2
model=$3
name=$4

if [ ! -n "$5" ] ;then
  folder=exp4
else
  folder=$5
fi

mkdir ./results/$folder
mkdir ./results/$folder/$target
mkdir ./results/$folder/$target/$model

CUDA_VISIBLE_DEVICES=$cuda python train_optim.py \
-cfg=$target/$model.yaml \
-s=./results/$folder/$target/$model/ \
-n=$model-$name-1 \
>./results/$folder/$target/$model/$model-$name-1.log

CUDA_VISIBLE_DEVICES=$cuda python train_optim.py \
-cfg=$target/$model.yaml \
-s=./results/$folder/$target/$model/ \
-n=$model-$name-2 \
>./results/$folder/$target/$model/$model-$name-2.log

CUDA_VISIBLE_DEVICES=$cuda python train_optim.py \
-cfg=$target/$model.yaml \
-s=./results/$folder/$target/$model/ \
-n=$model-$name-3 \
>./results/$folder/$target/$model/$model-$name-3.log