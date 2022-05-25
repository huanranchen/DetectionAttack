#CUDA_VISIBLE_DEVICES=3 python evaluate.py -i -o \
#-p ./results/coco/patch/1_90000_coco1.png \
#-cfg ./configs/coco1.yaml \
#-gt /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/labels \
#-dr /home/chenziyan/work/BaseDetectionAttack/data/INRIAPerson/Train/pos \
#-s /home/chenziyan/work/BaseDetectionAttack/data/inria/5-23

CUDA_VISIBLE_DEVICES=3 python evaluate.py -i -o \
-p ./results/coco/patch/1_90000_coco1.png \
-cfg ./configs/coco1.yaml \
-gt /home/chenziyan/work/BaseDetectionAttack/data/coco/labels \
-dr /home/chenziyan/work/BaseDetectionAttack/data/coco/train2017 \
-s /home/chenziyan/work/BaseDetectionAttack/data/coco/5-24