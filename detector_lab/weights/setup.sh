# Download weights for vanilla YOLOv3
wget -c "https://pjreddie.com/media/files/yolov3.weights" --header "Referer: pjreddie.com"
# # Download weights for tiny YOLOv3
wget -c "https://pjreddie.com/media/files/yolov3-tiny.weights" --header "Referer: pjreddie.com"
# Download weights for backbone network
wget -c "https://pjreddie.com/media/files/darknet53.conv.74" --header "Referer: pjreddie.com"

ln -s $pwd/darknet53.conv.74 ../yolov3/PyTorch_YOLOv3/weights/darknet53.conv.74
ln -s $pwd/yolov3.weights ../yolov3/PyTorch_YOLOv3/weights/yolov3.weights


ln -s $pwd/yolov4.conv.137.pth ../yolov4/Pytorch_YOLOv4/weight/yolov4.conv.137.pth
ln -s $pwd/yolov4.pth ../yolov4/Pytorch_YOLOv4/weight/yolov4.pth
ln -s $pwd/yolov4.weights ../yolov4/Pytorch_YOLOv4/weight/yolov4.weights


ln -s $pwd/yolov5s.pt ../yolov5/yolov5/weight/yolov5s.pt

# ssd
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth

# faster rcnn
wget http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
wget https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
ln -s $pwd/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth ../faster_rcnn/pytorch_faster_rcnn/weights/
ln -s $pwd/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth ../faster_rcnn/pytorch_faster_rcnn/weights/