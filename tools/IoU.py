import numpy as np


def cal_iou(bboxes, target):
    """Calculate the IoU between bboxes and the target

    :param bboxes: all the bounding boxes
    :type bboxes: numpy.ndarray
    :shape bboxes: (n,4)
    :param target: [description]
    :type target: numpy.ndarray
    :shape target: (4,)
    """
    bboxes = np.atleast_2d(bboxes)

    area_bboxes = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    area_target = (target[2] - target[0]) * (target[3] - target[1])

    iou_x1 = np.maximum(bboxes[:, 0], target[0])
    iou_y1 = np.maximum(bboxes[:, 1], target[1])
    iou_x2 = np.minimum(bboxes[:, 2], target[2])
    iou_y2 = np.minimum(bboxes[:, 3], target[3])

    iou_w = iou_x2 - iou_x1
    iou_h = iou_y2 - iou_y1
    area_iou = iou_w * iou_h

    return area_iou / (area_bboxes + area_target - area_iou)
    