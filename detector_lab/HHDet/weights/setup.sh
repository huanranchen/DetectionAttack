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