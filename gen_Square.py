#zfg 生成棋盘格

import cv2
import sys
import numpy as np

image = np.ones([1080, 1920, 3], np.uint8) * 255
x_nums = 14
y_nums = 7
square_pixel = 120  # 1080/9 = 120 pixels
x0 = square_pixel
y0 = square_pixel


def DrawSquare():
    flag = -1
    for i in range(y_nums):
        flag = 0 - flag
        for j in range(x_nums):
            if flag > 0:
                color = [0, 0, 0]
            else:
                color = [255, 255, 255]
            cv2.rectangle(image, (x0 + j * square_pixel, y0 + i * square_pixel),
                          (x0 + j * square_pixel + square_pixel, y0 + i * square_pixel + square_pixel), color, -1)
            flag = 0 - flag
    cv2.imwrite(r'E:\Zhang_Calibration\chess_map_14x7.bmp', image)


if __name__ == '__main__':
    DrawSquare()
