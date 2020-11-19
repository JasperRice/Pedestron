import argparse
import glob
import json
import os
import os.path as osp
import pdb
import socket
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

from IoU import cal_iou
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
    parser.add_argument('--image_save_path', type=str, default='', 
                        help='the path for output images')
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


def simple_visualization_and_sender(image, results, class_names, score_thr=0.9, model=None, poly=None, real_human_height=180, font_scale=1.5e-3, font_gap=40, box_thickness=2, socket_=None):
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
        original_labels = labels.copy()
        labels = labels[inds]
    if len(labels) > 0:
        ind_with_max_score = np.argmax(scores)
        label_with_max_score = original_labels[ind_with_max_score]
    
    bbox_color = (0, 255, 0)
    x_meter = y_meter = z_meter = ""
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

            x_deviation = int(real_distance_pixel *
                              (width-bbox_int[0]-bbox_int[2])/2)
            y_deviation = int(real_distance_pixel *
                              (height-bbox_int[1]-bbox_int[3])/2)
            z_deviation = int(predict_distance)
            cv2.putText(image, 'x - {:d} cm'.format(x_deviation), (bbox_int[2], bbox_int[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*height, (147, 20, 255), 2)
            cv2.putText(image, 'y - {:d} cm'.format(y_deviation), (bbox_int[2], bbox_int[1]+font_gap),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*height, (147, 20, 255), 2)
            cv2.putText(image, 'z - {:d} cm'.format(z_deviation), (bbox_int[2], bbox_int[1]+2*font_gap),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*height, (147, 20, 255), 2)

            current_bbox_area = (bbox_int[2] - bbox_int[0]) * (bbox_int[3] - bbox_int[1])
            if max_bbox_area < current_bbox_area:
                x_meter, y_meter, z_meter = str(x_deviation/100), str(y_deviation/100), str(z_deviation/100) # x, y, z in meter

        cv2.rectangle(image, left_top, right_bottom, bbox_color, thickness=box_thickness)

    deviation = '|'.join([x_meter, y_meter, z_meter])
    try:
        socket_.conn.send(bytes(deviation, encoding='utf-8'))
    except:
        # print("Failed to send the position.")
        pass

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


def run_detector_on_webcam(predict_model=None, poly=None, threshold=0.95, webcam_index=0, crop=0):
    args = parse_args()
    model = init_detector(args.config, args.checkpoint,
                          device=torch.device('cuda:0'))
    cv2.namedWindow("Human_Detector", 0)
    cv2.resizeWindow("Human_Detector", 1920//2, 1080//2)
    cv2.moveWindow("Human_Detector", -100, -100)
    try: receiver = Receiver(webcam_index)
    except:
        pass
    
    flag = False
    old_flag = False
    count = 0
    for image in receiver:
        if crop > 0:
            _, width, _ = image.shape
            d_width = (1-crop)*width//2
            image = image[:, int(width/2-d_width):int(width/2+d_width)]
        results = inference_detector(model, image)
        image = simple_visualization_and_sender(
            image, results, model.CLASSES, model=predict_model, poly=poly, score_thr=threshold)
        # cv2.resize(image, (1920//3, 1080//3))
        cv2.imshow('Human_Detector', image)
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            old_flag = flag
            flag = not flag
            if flag: print("Start to record ...")
            if old_flag and not flag: print("Stop recording.")

        if args.image_save_path != '' and flag:
            filename = '{}/{}.jpg'.format(args.image_save_path, str(count).zfill(5))
            cv2.imwrite(filename, image)
            print("Saving {} ...".format(filename))
            count += 1

    cv2.destroyAllWindows()


def run_detector_on_tcp_webcam(predict_model=None, poly=None, threshold=0.99, crop=0):
    args = parse_args()
    model = init_detector(args.config, args.checkpoint,
                          device=torch.device('cuda:0'))

    # webcam_recevier = Tcp_Receiver('192.168.8.142', 8020, 16)
    webcam_recevier = Tcp_Receiver('192.168.31.101', 8020, 16)
    for image in webcam_recevier:
        if crop > 0:
            _, width, _ = image.shape
            d_width = (1-crop)*width//2
            image = image[:, int(width/2-d_width):int(width/2+d_width)]

        results = inference_detector(model, image)
        image = simple_visualization_and_sender(
            image, results, model.CLASSES, model=predict_model, poly=poly, socket_=webcam_recevier, score_thr=threshold)
        cv2.imshow('SERVER', image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Quit the server.")
            break
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
    # run_detector_on_video(predict_model=model, poly=poly)
    run_detector_on_webcam(predict_model=model, poly=poly, threshold=0.85, webcam_index=0, crop=1/2)
    # run_detector_on_tcp_webcam(predict_model=model, poly=poly, threshold=0.85, crop=1/2)
