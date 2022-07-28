import numpy as np
import torch
import torchvision


def inter_nms(all_predictions, conf_thres=0.25, iou_thres=0.45):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    max_det = 300  # maximum number of detections per image
    out = []
    for predictions in all_predictions:
        # for each img in batch
        # print('pred', predictions.shape)
        if not predictions.shape[0]:
            out.append(predictions)
            continue
        if type(predictions) is np.ndarray:
            predictions = torch.from_numpy(predictions)

        if predictions.ndim == 1:
            predictions = predictions.unsqueeze(0)
        # print(predictions.shape[0])
        scores = predictions[:, 4]
        i = scores > conf_thres

        # filter with conf threshhold
        boxes = predictions[i, :4]
        scores = scores[i]

        # filter with iou threshhold
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        # print('i', predictions[i].shape)
        out.append(predictions[i])
    return out