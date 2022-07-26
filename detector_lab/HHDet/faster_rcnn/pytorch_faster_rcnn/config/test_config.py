class Config:
    model_weights = "/home/chenziyan/work/BaseDetectionAttack/detector_lab/HHDet/weights/faster_rcnn/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    image_path = "/home/chenziyan/work/BaseDetectionAttack/data/coco/devKit/train2017"
    gpu_id = '2'
    num_classes = 80 + 1
    data_root_dir = "/home/chenziyan/work/BaseDetectionAttack/data/coco/devKit/"


test_cfg = Config()
