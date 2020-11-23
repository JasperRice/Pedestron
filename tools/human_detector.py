import os
import pdb
import sys
from multiprocessing import Process, Queue

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.insert(0, '/home/midea/Documents/Pedestron')
sys.path.insert(0, '/home/midea/Documents/Pedestron/tools')

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
                 human_cm=180,
                 distance_predictor=None,
                 ):
        self.model = init_detector(config, checkpoint, device)
        self.threshold = threshold

        self.human_cm = human_cm
        self.distance_predictor = distance_predictor

        self.ind_target = 0
        self.target_bbox = None

        # TODO: Should I keep the old result or not?
        self.old_bboxes = None
        self.old_scores = None

        self.bboxes = None
        self.scores = None

    def __call__(self, image):
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
        self.scores = scores[index]
        self.bboxes = bboxes[index, :].astype(np.int32)

    def update(self, image, connect=None):
        height, width, _ = image.shape
        self(image)

        if len(self.bboxes) > 0 and self.bboxes is not None:
            if self.target_bbox is not None:
                ious = cal_iou(self.bboxes, self.target_bbox)
                self.ind_target = np.argmax(ious)
            else:
                area_bboxes = (
                    self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])
                self.ind_target = np.argmax(area_bboxes)
            self.target_bbox = self.bboxes[self.ind_target]
        else:
            # TODO: Should I do something here?
            # When there's no bounding boxes with score higher
            # than the threshold is found
            self.target_bbox = None

        if self.distance_predictor is not None and connect is not None:
            x = y = z = ''
            if self.distance_predictor is not None and self.target_bbox is not None:
                human_px = self.target_bbox[3] - self.target_bbox[1]
                cm_per_pixel = self.human_cm / human_px
                inverse_ratio = np.array(height / human_px).reshape(1, -1)
                predict_distance = self.distance_predictor.predict(inverse_ratio)[0]
                x = cm_per_pixel * (width - self.target_bbox[0] - self.target_bbox[2]) / 2 / 100
                y = cm_per_pixel * (height - self.target_bbox[1] - self.target_bbox[3]) / 2 / 100
                z = predict_distance / 100
                position = list(map(lambda a: '{:.3}'.format(a), [x, y, z]))
            else:
                position = [x, y, z]
            self.send(connect, position)

    def send(self, connect, position):
        position = '|'.join(position)
        connect.send(bytes(position, encoding='utf-8'))

    def draw(self, image):
        for i, (bbox, score) in enumerate(zip(self.bboxes, self.scores)):
            pass
        return image


if __name__ == "__main__":
    from webcam_wrapper import Receiver
    receiver = Receiver(0)
    
    config = 'configs/pascal_voc/ssd300_voc.py'
    checkpoint = '/data/sifan/model-zoo/pedestron/ssd300_voc_vgg16_caffe_240e_20190501-7160d09a.pth'
    detector = HumanDetector(config, checkpoint)

    for image in receiver:
        detector.update(image)
