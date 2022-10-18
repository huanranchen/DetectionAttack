
cuda=$1
target=$2
model=$3

bash test.sh $cuda \
exp4/$target/$model/$model-$target-scale \
exp4/$target/$model/ \
"eval/coco80 eval/coco91" \
"Test"