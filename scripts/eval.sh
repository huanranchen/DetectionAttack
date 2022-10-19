
cuda=$1
target=$2
model=$3
name=$4

bash test.sh $cuda \
exp4/$target/$model/$model-$name \
exp4/$target/$model/ \
"eval/coco80 eval/coco91" \
"Test"