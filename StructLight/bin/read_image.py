#!/usr/bin/python3
# -*- coding: utf-8 -*-
__Author__ = 'csd'
__version__ = "1.0"
__data__ = "2019.04.17"

import os
import sys
import DLP4500
import pymvcam
import cv2
import numpy as np
import time
from .GrayPattern import GrayPattern,showPointCloud

FIRST_PATTERN_INDEX = 3
LAST_PATTERN_INDEX = 47

def main():
    DLP = DLP4500.DLP4500API()
    DLP.setLEDCurrentMode(1)
    DLP.setLEDCurrentValue(40, 40, 40)
    DLP.setLEDSelection(0, 0, 1)
    gray_pattern = GrayPattern("../res/input/Stereo")
    cam1 = pymvcam.MVCam(acFriendlyName='Cam4')
    cam1.setExposureTime(5000)
    cam2 = pymvcam.MVCam(acFriendlyName='Cam6')
    cam2.setExposureTime(5000)
    cam1.start()
    cam2.start()
    print("Enter Esc to continue capture image")
    t1 = time.time()
    while True:
        cam1.softTrigger()
        image1 = cam1.readImage(timeout_ms=2000)
        cv2.imshow("image1", image1)
        cam2.softTrigger()
        image2 = cam2.readImage(timeout_ms=2000)
        cv2.imshow("image2", image2)
        if time.time() - t1 > 2:
            break
    # cam1.cleanBuffer()
    # cam2.cleanBuffer()
    image_left = []
    image_right = []
    for i in range(FIRST_PATTERN_INDEX, LAST_PATTERN_INDEX):
        DLP.loadImageFromFlash(i)
        time.sleep(0.1)
        cam1.softTrigger()
        time.sleep(0.1)
        image1 = cam1.readImage()
        # cv2.imshow("image1", image1)
        image_left.append(image1.copy())
        cam2.softTrigger()
        time.sleep(0.1)
        image2 = cam2.readImage()
        # cv2.imshow("image2", image2)
        image_right.append(image2.copy())
        cv2.waitKey(20)
        # cv2.imwrite("../res/output/image/cam4/image_" + str(i) + ".png", image2)
        # cv2.imwrite("../res/output/image/cam6/image_" + str(i) + ".png", image1)
    print("image capture success, calculating point cloud...")
    disp, point_cloud = gray_pattern.run(image_left, image_right)
    showPointCloud(point_cloud)


if __name__ == "__main__":
    main()
