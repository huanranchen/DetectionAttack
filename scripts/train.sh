
cuda=7
model=v4tiny
target=baseline

mkdir ./results/exp4/$target/$model
CUDA_VISIBLE_DEVICES=$cuda nohup python train_optim.py \
-cfg=$target/$model.yaml \
-s=./results/exp4/$target/$model/ \
-n=$model-$target-scale \
>./results/exp4/$target/$model/$model-$target-scale.log 2>&1 &