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


def main(rotate=-90, downsample=4):
    video = cv2.VideoCapture(
        '/data/sifan/videos/original/follow/webcam.mp4')
    rval, frame = video.read()
    if rval:
        height, width, _ = frame.shape
        frameSize = (width//downsample, height//downsample)
    c = 0
    while rval:
        c += 1
        if c % 3 == 1:
            if rval:
            frame = cv2.resize(frame, frameSize)
            frame = np.rot90(frame, rotate//90)
            cv2.imwrite(
                '/data/sifan/images/results/follow/webcam/{}.jpg'.format(c), frame)
            cv2.waitKey(1)
        else:
                break
        else:
            rval, frame = video.read()
    video.release()


if __name__ == "__main__":
    main(rotate=-90, downsample=4)
