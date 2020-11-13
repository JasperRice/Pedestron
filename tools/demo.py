import argparse
import glob
import json
import os
import os.path as osp
import sys
import time

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import cv2
import mmcv
import numpy as np
import torch
from mmdet.apis import inference_detector, init_detector, show_result
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_img_dir', type=str,
                        help='the dir of input images')
    parser.add_argument('output_dir', type=str,
                        help='the dir for result images')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mean_teacher', action='store_true',
                        help='test the mean teacher pth')
    parser.add_argument('--image_type', type=str, default='jpg')
    args = parser.parse_args()
    return args


def mock_detector(model, image_name, output_dir, save_bbox=False, predict_model=None, poly=None):
    image = cv2.imread(image_name)
    results = inference_detector(model, image)
    basename = os.path.basename(image_name).split('.')[0]
    result_name = basename + "_result.jpg"
    result_name = os.path.join(output_dir, result_name)
    show_result(image, results, model.CLASSES, score_thr=0.8,
                out_file=result_name, save_bbox=save_bbox, predict_model=predict_model, poly=poly)


def create_base_dir(dest):
    basedir = os.path.dirname(dest)
    if not os.path.exists(basedir):
        os.makedirs(basedir)


def run_detector_on_dataset(save_bbox=False, predict_model=None, poly=None):
    args = parse_args()
    input_dir = args.input_img_dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(input_dir)
    eval_imgs = glob.glob(os.path.join(input_dir, '*.'+args.image_type))
    print(eval_imgs)

    model = init_detector(
        args.config, args.checkpoint, device=torch.device('cuda:0'))

    # prog_bar = mmcv.ProgressBar(len(eval_imgs), start=True)
    start = time.time()
    for im in eval_imgs:
        detections = mock_detector(
            model, im, output_dir, save_bbox=save_bbox, predict_model=predict_model, poly=poly)
        # prog_bar.update()
    end = time.time()
    print("Total time: {:.3f} s".format(end-start))


if __name__ == '__main__':
    model = LinearRegression()
    poly = None
    # poly = PolynomialFeatures(degree=2)
    original_height_pixel = 1080
    # with open('/home/sifan/Documents/Pedestron/height-distance-webcam.txt') as file:
    #     lines = file.readlines()
    #     X = np.array(list(map(float, [line.split()[0]
    #                                   for line in lines]))).reshape(-1, 1)
    #     X = original_height_pixel / X
    #     if poly != None:
    #         X = poly.fit_transform(X)
    #     Y = np.array(list(map(float, [line.split()[1] for line in lines])))
    #     model.fit(X, Y)

    run_detector_on_dataset(save_bbox=False, predict_model=None, poly=None)
