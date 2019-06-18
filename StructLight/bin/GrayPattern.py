#!/usr/bin/python3
# -*- coding: utf-8 -*-
__Author__ = 'csd'
__version__ = "1.0"
__data__ = "2019.04.11"

import os
import sys
import re
import numpy as np
import cv2
import open3d
import copy


def pointCloudCut(src_pc, lower_mm, upper_mm, axis=2):
    point_cloud_nx3 = src_pc[src_pc[:, axis] > lower_mm]
    point_cloud_nx3 = point_cloud_nx3[point_cloud_nx3[:, axis] < upper_mm]
    return point_cloud_nx3

def showPointCloud(poindCloud):
        pcd = open3d.PointCloud()
        pcd.points = open3d.Vector3dVector(poindCloud)
        open3d.draw_geometries([pcd])
        open3d.write_point_cloud("../res/output/point_cloud.pcd", pcd)

def findNumber(str, numMatch=r"(\d*)\."):
    return int(re.findall(numMatch, str)[0])


def pickupImgFiles(filePath, pickFileType=['.png', '.jpeg', '.bmp', '.jpg', '.gif'], numMatch=r"(\d*)\."):
    AbsPath = os.path.abspath(filePath)
    FileList = os.listdir(AbsPath)
    pickFileName = []
    for name in FileList:
        for type in pickFileType:
            if type in name:
                try:
                    findNumber(name, numMatch)
                    pickFileName.append(name)
                except:
                    pass
    pickFileName = sorted(pickFileName, key=lambda a: findNumber(a, numMatch))
    return pickFileName


def readImageFromPath(image_path):
    image_file_name = pickupImgFiles(image_path)
    image = []
    for image_file in image_file_name:
        image.append(cv2.imread(os.path.join(image_path, image_file), 0))
    return image


class GrayPattern():
    def __init__(self, stereoCaliPath, width=912, height=1140):
        self.__pattern_width = width
        self.__pattern_height = height
        self.__pattern_num = 0
        self.__image_left = []
        self.__image_right = []
        self.__capture = []
        self.__white_image = []
        self.__black_image = []
        self.__pattern_image = []
        self.__calibrate_data = {}
        self.initGenerater()
        self.__loadCalibrateData(stereoCaliPath)

    def initGenerater(self):
        self.__pattern_generate = cv2.structured_light_GrayCodePattern.create(width=self.__pattern_width, height=self.__pattern_height)
        self.__pattern_num = self.__pattern_generate.getNumberOfPatternImages()

    def preTreatmentImage(self, imgLeft, imgRight):
        self.__image_left = copy.deepcopy(imgLeft)
        self.__image_right = copy.deepcopy(imgRight)
        self.__white_image.append(self.__image_left[-1])
        self.__white_image.append(self.__image_right[-1])
        self.__black_image.append(self.__image_left[-2])
        self.__black_image.append(self.__image_right[-2])

    def generatePatternImage(self):
        ret, pattern = self.__pattern_generate.generate()
        white = np.ones((self.__pattern_width, self.__pattern_height), dtype=np.uint8)
        black = np.zeros((self.__pattern_width, self.__pattern_height), dtype=np.uint8)

        black, white = self.__pattern_generate.getImagesForShadowMasks(white, black)
        pattern.append(white)
        pattern.append(black)
        return pattern

    def __loadCalibrateData(self, caliPath):
        self.__calibrate_data = {
            "camera_matrix_left": np.loadtxt(os.path.join(caliPath, "CameraMatrixL.txt")),
            "camera_matrix_right": np.loadtxt(os.path.join(caliPath, "CameraMatrixR.txt")),
            "distorsion_left": np.loadtxt(os.path.join(caliPath, "DistCoeffsL.txt")),
            "distorsion_right": np.loadtxt(os.path.join(caliPath, "DistCoeffsR.txt")),
            "image_size": tuple(np.loadtxt(os.path.join(caliPath, "ImgSize.txt")).astype(int)),
            "R": np.loadtxt(os.path.join(caliPath, "R.txt")),
            "t": np.loadtxt(os.path.join(caliPath, "t.txt")),
        }
        RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(cameraMatrix1=self.__calibrate_data["camera_matrix_left"],
                                                    distCoeffs1=self.__calibrate_data["distorsion_left"],
                                                    cameraMatrix2=self.__calibrate_data["camera_matrix_right"],
                                                    distCoeffs2=self.__calibrate_data["distorsion_right"],
                                                    imageSize=self.__calibrate_data["image_size"],
                                                    R=self.__calibrate_data["R"],
                                                    T=self.__calibrate_data["t"])
        self.__calibrate_data["Q"] = Q
        map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix=self.__calibrate_data["camera_matrix_left"],
                                                   distCoeffs=self.__calibrate_data["distorsion_left"],
                                                   R=RL, newCameraMatrix=PL,
                                                   size=self.__calibrate_data["image_size"],
                                                   m1type=cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix=self.__calibrate_data["camera_matrix_right"],
                                                   distCoeffs=self.__calibrate_data["distorsion_right"],
                                                   R=RR, newCameraMatrix=PR,
                                                   size=self.__calibrate_data["image_size"],
                                                   m1type=cv2.CV_32FC1)
        self.__calibrate_data["map1x"] = map1x
        self.__calibrate_data["map1y"] = map1y
        self.__calibrate_data["map2x"] = map2x
        self.__calibrate_data["map2y"] = map2y


    def rectifyImage(self):
        self.__white_image[0] = cv2.remap(self.__white_image[0], self.__calibrate_data["map2x"], self.__calibrate_data["map2y"], cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        self.__white_image[1] = cv2.remap(self.__white_image[1], self.__calibrate_data["map1x"], self.__calibrate_data["map1y"], cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        self.__black_image[0] = cv2.remap(self.__black_image[0], self.__calibrate_data["map2x"], self.__calibrate_data["map2y"], cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        self.__black_image[1] = cv2.remap(self.__black_image[1], self.__calibrate_data["map1x"], self.__calibrate_data["map1y"], cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)

        for i in range(0, self.__pattern_num):
            self.__image_left[i] = cv2.remap(self.__image_left[i], self.__calibrate_data["map2x"], self.__calibrate_data["map2y"], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
            self.__image_right[i] = cv2.remap(self.__image_right[i], self.__calibrate_data["map1x"], self.__calibrate_data["map1y"], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

        image_left = self.__image_left[:-2]
        image_right = self.__image_right[:-2]
        self.__capture.append(image_left)
        self.__capture.append(image_right)

    def generatePointCloud(self):
        ret, disp_map = self.__pattern_generate.decode(self.__capture,
                                                   blackImages=self.__black_image,
                                                   whiteImages=self.__white_image)
        if ret:
            disp_map = np.float32(disp_map)
            point_cloud = cv2.reprojectImageTo3D(disp_map, self.__calibrate_data["Q"], handleMissingValues=True, ddepth=-1)
            point_cloud_nx3 = point_cloud[np.where(point_cloud[:, :, 2] < 999)]
            return disp_map, point_cloud_nx3
        return None, None

    def run(self, imgLeft, imgight):
        self.preTreatmentImage(imgLeft, imgight)
        self.rectifyImage()
        disp, point_cloud = self.generatePointCloud()
        return disp, point_cloud



def main():
    calibrate_info_path = "../res/input/data1/Stereo"
    image_cam_left = "../res/input/data1/coin2/Cam4"
    image_cam_right = "../res/input/data1/coin2/Cam6"
    image_left = readImageFromPath(image_cam_left)
    image_right = readImageFromPath(image_cam_right)
    my_gray_pattern = GrayPattern(calibrate_info_path)
    disp, point_cloud = my_gray_pattern.run(image_left, image_right)
    if point_cloud is not None:
        showPointCloud(point_cloud)



if __name__ == "__main__":
    main()
