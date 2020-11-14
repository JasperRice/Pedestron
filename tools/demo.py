import argparse
import glob
import json
import os
import os.path as osp
import sys
import time

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import mmcv
import numpy as np
import torch
from cv2 import cv2
from mmdet.apis import inference_detector, init_detector, show_result
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from webcam_wrapper import Receiver, Tcp_Receiver


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input_img_dir', type=str,
                        help='the dir of input images')
    parser.add_argument('--output_dir', type=str,
                        help='the dir for result images')
    parser.add_argument('--video_path', type=str,
                        help='the path for input video')
    parser.add_argument('--image_type', type=str, default='jpg')
    parser.add_argument('--save_bbox', type=bool, default=False)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true',
                        help='test the mean teacher pth')
    args = parser.parse_args()
    return args


def simple_visualization(image, results, class_names, score_thr=0.9, model=None, poly=None, real_human_height=180, font_scale=7e-4):
    assert isinstance(class_names, (tuple, list))
    
    if isinstance(results, tuple):
        bbox_result, segm_result = results
    else:
        bbox_result, segm_result = results, None
    segm_result = None
    bboxes = np.vstack(bbox_result)

    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
    ind_with_max_score = scores.index(max(scores))

    bbox_color = (0, 255, 0)

    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        height, width, _ = image.shape

        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])

        if model != None:
            human_height = max(
                bbox_int[2] - bbox_int[0], bbox_int[3] - bbox_int[1])
            real_distance_pixel = real_human_height / human_height
            inverse_ratio = np.array(height / human_height).reshape(1, -1)
            if poly != None:
                inverse_ratio = poly.fit_transform(inverse_ratio)
            predict_distance = model.predict(inverse_ratio)[0]

            x_deviation = int(real_distance_pixel*(width-bbox_int[0]-bbox_int[2])/2)
            y_deviation = int(real_distance_pixel*(height-bbox_int[1]-bbox_int[3])/2)
            z_deviation = int(predict_distance)
            cv2.putText(image, 'x - {:d} cm'.format(x_deviation), (bbox_int[2], bbox_int[1]+0),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*height, (147,20,255), 2)
            cv2.putText(image, 'y - {:d} cm'.format(y_deviation), (bbox_int[2], bbox_int[1]+60),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*height, (147,20,255), 2)
            cv2.putText(image, 'z - {:d} cm'.format(z_deviation), (bbox_int[2], bbox_int[1]+120),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*height, (147,20,255), 2)

            if i == ind_with_max_score:
                pass
                # Send x, y, z

        cv2.rectangle(image, left_top, right_bottom, bbox_color, thickness=1)

    return image


def mock_detector(model, image_name, output_dir, save_bbox=False, predict_model=None, poly=None):
    image = cv2.imread(image_name)
    results = inference_detector(model, image)
    basename = os.path.basename(image_name).split('.')[0]
    result_name = basename + "_result.jpg"
    result_name = os.path.join(output_dir, result_name)
    show_result(image, results, model.CLASSES, score_thr=0.8,
                out_file=result_name, save_bbox=save_bbox, predict_model=predict_model, poly=poly)  # Default: score_thr=0.8


def create_base_dir(dest):
    basedir = os.path.dirname(dest)
    if not os.path.exists(basedir):
        os.makedirs(basedir)


def run_detector_on_dataset(predict_model=None, poly=None):
    args = parse_args()
    input_dir = args.input_img_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(input_dir)
    eval_imgs = glob.glob(os.path.join(input_dir, '*.'+args.image_type))
    # print(eval_imgs)

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    prog_bar = mmcv.ProgressBar(len(eval_imgs))
    for im in eval_imgs:
        detections = mock_detector(
            model, im, output_dir, save_bbox=args.save_bbox, predict_model=predict_model, poly=poly)
        prog_bar.update()


def run_detector_on_video(predict_model=None, poly=None):
    args = parse_args()
    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    cap = cv2.VideoCapture(args.video_path)
    ret, frame = cap.read()
    while ret:
        results = inference_detector(model, frame)
        show_result(frame, results, model.CLASSES, score_thr=0.8,
                    out_file=None, save_bbox=args.save_bbox, predict_model=predict_model, poly=poly)  # Default: score_thr=0.8
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


def run_detector_on_webcam(predict_model=None, poly=None):
    args = parse_args()
    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    webcam_recevier = Tcp_Receiver('192.168.8.142', 8020, 16)
    for frame in webcam_recevier:
        results = inference_detector(model, frame)
        show_result(frame, results, model.CLASSES, score_thr=0.8,
                    out_file=None, save_bbox=args.save_bbox, predict_model=predict_model, poly=poly)  # Default: score_thr=0.8
    cv2.destroyAllWindows()


def run_detector_on_tcp_webcam(predict_model=None, poly=None):
    args = parse_args()
    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    while ret:
        results = inference_detector(model, frame)
        show_result(frame, results, model.CLASSES, score_thr=0.8,
                    out_file=None, save_bbox=args.save_bbox, predict_model=predict_model, poly=poly)  # Default: score_thr=0.8
        ret, frame = cap.read()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    model = LinearRegression()
    poly = None
    original_height_pixel = 1080
    with open('height-distance-webcam.txt', 'r') as file:
        lines = file.readlines()
        X = np.array(list(map(float, [line.split()[0]
                                      for line in lines]))).reshape(-1, 1)
        X = original_height_pixel / X
        if poly != None:
            X = poly.fit_transform(X)
        Y = np.array(list(map(float, [line.split()[1] for line in lines])))
        model.fit(X, Y)

    # run_detector_on_dataset(predict_model=model, poly=poly)
    run_detector_on_video(predict_model=model, poly=poly)
