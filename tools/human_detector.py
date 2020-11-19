import os
import pdb
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../'))

import numpy as np
from cv2 import cv2 as cv
from mmdet.apis import inference_detector, init_detector

from colors import BGR
from IoU import cal_iou


class HumanDetector():

    def __init__(self, config,
                 checkpoint=None,
                 device='cuda:0',
                 threshold=0.85,
                 bbox_color=BGR['DarkSeaGreen1'],
                 track_bbox_color=BGR['Firebrick1'],
                 thickness=3,
                 ):
        self.model = init_detector(config, checkpoint, device)
        self.threshold = threshold

        self.target_bbox = None

        self.old_bboxes = None
        self.old_scores = None

        self.bboxes = None
        self.scores = None

    def update(self, image):
        result = inference_detector(self.model, image)
        if isinstance(result, tuple):
            bboxes, _ = result
        else:
            bboxes = result
        bboxes = np.vstack(bboxes)

        scores = bboxes[:, -1]
        bboxes = bboxes[:, 0:4]
        index = scores > self.threshold
        
        self.old_scores = self.scores
        self.old_bboxes = self.bboxes
        self.scores = scores[index, :]
        self.bboxes = bboxes[index, :].astype(np.int32)

        if self.target_bbox is not None:
            pass

    def draw(self, image):
        pass

    def send(self):
        pass
