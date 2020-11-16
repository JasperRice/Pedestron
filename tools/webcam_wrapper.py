import socket
import sys
import time

import numpy as np
from cv2 import cv2 as cv


class Receiver(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:
            self.file_name = int(file_name)
        except ValueError:
            pass
        self.cap = cv.VideoCapture(self.file_name)

    def set_resolution(self, width, height):
        self.cap.set(3, width)
        self.cap.set(4, height)

    def __iter__(self):
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img


class Tcp_Receiver(object):
    def __init__(self, ip, port, count, record_time=False):
        self.address = (ip, port)
        self.count = count
        self.if_record_times = record_time

    def __iter__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(self.address)
        try:
            self.socket.listen(1)
            print("Start listening ...")
        except:
            print("Fail to initialize reveiver.")

        self.conn, self.addr = self.socket.accept()
        print("Being connected from: {}.".format(self.addr))

        return self

    def __next__(self):
        if self.if_record_times:
            self.start_time_length = Tcp_Receiver.recv_all(self.conn, 20)
            self.start_time_string_data = float(self.conn.recv(
                int(str(self.start_time_length, encoding="utf-8"))))
        
        self.length = Tcp_Receiver.recv_all(self.conn, self.count)
        self.string_data = Tcp_Receiver.recv_all(self.conn, int(self.length))
        self.data = np.frombuffer(self.string_data, np.uint8)
        img = cv.imdecode(self.data, cv.IMREAD_COLOR)

        return img

    @staticmethod
    def recv_all(sock, count):
        """[summary]

        :param sock: [description]
        :type sock: [type]
        :param count: 接收的最大数据量
        :type count: int
        :return: [description]
        :rtype: [type]
        """
        buf = b''  # buf是一个byte类型
        while count:
            # 接受TCP套接字的数据。数据以字符串形式返回。
            newbuf = sock.recv(count)
            if not newbuf:
                return None
            buf += newbuf
            count -= len(newbuf)
        return buf


if __name__ == "__main__":
    # Test tcp_receiver
    # receiver = Tcp_Receiver('192.168.8.142', 8020, 16)

    # Test receiver
    receiver = Receiver(0)
    
    for image in receiver:
        cv.imshow('SERVER', image)
        cv.waitKey(1)
    cv.destroyAllWindows()

