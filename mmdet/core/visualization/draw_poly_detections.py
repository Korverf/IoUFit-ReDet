import numpy as np
import cv2
import random


def draw_poly_detections(detections, img=None, showStart=False, colormap=None):
    if img is None:
        img = 255 * np.ones((1024, 1024, 3), dtype='uint8')
    #color_white = (255, 255, 255)

    if colormap is None:
        color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
    else:
        color = colormap
    dets = detections

    # 绘制框
    if len(dets) > 0:
        for det in dets:
            bbox = det
            bbox = list(map(int, bbox))
            if showStart:# 在第一个坐标绘制一个小圆点
                cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
            #绘制边框
            for i in range(3):
                cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]),
                            color=color, thickness=1,lineType=cv2.LINE_AA)
            cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color, thickness=1, lineType=cv2.LINE_AA)
    return img