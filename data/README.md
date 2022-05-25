#Evaluation of universal patches

### mAP.py
[mAP computation](https://github.com/Cartucho/mAP)


## Label process
### xx_process.txt

xx_precess.py is for processing annotations of the dataset into a standard yolo-style label format as:

```
cls_name x_min y_min x_max y_max

person 0.1345 0.4567 0.2456 0.9876
```

where cls_number is the index of the label in the class name file, and the xyxy of the bbox is scale into [0, 1].