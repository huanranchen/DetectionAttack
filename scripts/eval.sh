
cuda=$1
target=$2
model=$3

bash test.sh 2 \
exp4/$target/$model/$model-$target-scale \
exp4/$target/$model/ \
"eval/coco80 eval/coco91" \
"Test"