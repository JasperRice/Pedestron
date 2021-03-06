import cv2
import numpy as np
from mmcv.image import imread, imwrite

from .color import color_val


def imshow(img, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    cv2.waitKey(wait_time)


def imshow_bboxes(img,
                  bboxes,
                  colors='green',
                  top_k=-1,
                  thickness=1,
                  show=True,
                  win_name='',
                  wait_time=0,
                  out_file=None):
    """Draw bboxes on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (list or ndarray): A list of ndarray of shape (k, 4).
        colors (list[str or tuple or Color]): A list of colors.
        top_k (int): Plot the first k bboxes only if set positive.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.
    """
    img = imread(img)

    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        _bboxes = _bboxes.astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes[j, 0], _bboxes[j, 1])
            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
            cv2.rectangle(
                img, left_top, right_bottom, colors[i], thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color='green',
                      text_color='green',
                      thickness=1,
                      font_scale=7e-4,
                      font_thickness_scale=2,
                      real_human_height=180,
                      show=True,
                      win_name='',
                      wait_time=0,
                      out_file=None,
                      save_bbox=False,
                      predict_model=None,
                      poly=None):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        real_human_height: The height of the human in real world.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
        save_bbox: If save the height of the bounding box
        predict_model: The model to predict the distance based on the height of the bbox.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5

    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    for bbox, label in zip(bboxes, labels):
        height, width, _ = img.shape
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else 'cls {}'.format(label)
        if len(bbox) > 4:
            label_text += '|{:.02f}'.format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1]-int(0.004*height)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale*height, text_color, font_thickness_scale*1)  # Default: font_scale

        if save_bbox == True:
            with open('/home/sifan/Documents/Pedestron/height-distance-webcam.txt', 'a+') as file:
                human_height = max(
                    bbox_int[2] - bbox_int[0], bbox_int[3] - bbox_int[1])
                distance = int(((out_file.split('.')[-2]).split('/')[-1]).split('_')[0])
                file.writelines('{}\t{}\n'.format(human_height, distance))

        if predict_model != None:
            human_height = max(
                bbox_int[2] - bbox_int[0], bbox_int[3] - bbox_int[1])
            real_distance_pixel = real_human_height / human_height

            # [TEST] Print the heigth of the image and the height of the bbox
            # print(height, human_height)

            inverse_ratio = np.array(height / human_height).reshape(1, -1)
            if poly != None:
                inverse_ratio = poly.fit_transform(inverse_ratio)
            predict_distance = predict_model.predict(inverse_ratio)[0]

            cv2.putText(img, 'x - {:d} cm'.format(int(real_distance_pixel*(width-bbox_int[0]-bbox_int[2])/2)), (bbox_int[2], bbox_int[1]+0),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*height, (147,20,255), 2)
            cv2.putText(img, 'y - {:d} cm'.format(int(real_distance_pixel*(height-bbox_int[1]-bbox_int[3])/2)), (bbox_int[2], bbox_int[1]+60),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*height, (147,20,255), 2)
            cv2.putText(img, 'z - {:d} cm'.format(int(predict_distance)), (bbox_int[2], bbox_int[1]+120),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale*height, (147,20,255), 2)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)
