
cuda=$1
target=$2
model=$3
name=$4

if [ $model != "" ] ;then
  name="-"$name
fi

if [ ! -n "$5" ] ;then
  folder=exp4
else
  folder=$5
fi

bash test.sh $cuda \
$folder/$target/$model/$model$name \
$folder/$target/$model/ \
"eval/coco80 eval/coco91" \
"Test"