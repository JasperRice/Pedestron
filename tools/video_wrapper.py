import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


def main():
    vc = cv2.VideoCapture(
        '/home/sifan/Documents/Pedestron/demo/video/office_robot_follow_02.mp4')
    rval = vc.isOpened()
    c = 0
    while rval:
        c += 1
        rval, frame = vc.read()
        if rval:
            frame = np.rot90(frame, -1)
            cv2.imwrite(
                '/home/sifan/Documents/Pedestron/demo/video/office_robot_follow_02/office_robot_follow_02_{}.jpg'.format(c), frame)
            cv2.waitKey(1)
        else:
            break
    vc.release()


if __name__ == "__main__":
    main()
