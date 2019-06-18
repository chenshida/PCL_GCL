#!/usr/bin/python3
# -*- coding: utf-8 -*-
__Author__ = 'csd'
__version__ = "1.0"
__data__ = "2019.04.01"

import os
import sys
import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d as p3d

def drawPoint(pointCloud_3xn):
    fig = plt.figure()
    ax = p3d.Axes3D(fig)


    x = pointCloud_3xn[:, 0]
    y = pointCloud_3xn[:, 1]
    z = pointCloud_3xn[:, 2]

    ax.scatter(x, y, z, c='r')
    plt.show()

def planeFitBySVD(point_cloud):
    xyz_mean = [np.mean(point_cloud[:, 0]), np.mean(point_cloud[:, 1]), np.mean(point_cloud[:, 2])]
    print("xyz_mean: ", xyz_mean)
    point_vector = point_cloud - xyz_mean
    U, S, V = np.linalg.svd(point_vector)

    V = V.T
    a = V[0, 2]
    b = V[1, 2]
    c = V[2, 2]

    # print("a: ", a)
    # print("b: ", b)
    # print("c: ", c)

    d = np.dot([a, b, c], xyz_mean) * (-1)
    # print("d: ", d)
    return [a, b, c, d]

def planeEvaluate(point_cloud_nx3, plane_func_1x4):
    dist = []
    sqrt_val = np.sqrt(plane_func_1x4[0] * plane_func_1x4[0] + plane_func_1x4[1] * plane_func_1x4[1] + plane_func_1x4[2] * plane_func_1x4[2])
    for point in point_cloud_nx3:
        dist_item = abs((plane_func_1x4[0] * point[0] + plane_func_1x4[1] * point[1] + plane_func_1x4[2] * point[2] + plane_func_1x4[3]) / sqrt_val)
        dist.append(dist_item)
    dist_min = min(dist)
    dist_max = max(dist)
    dist_aver = np.mean(dist)
    dist_std = np.std(dist)
    return dist, dist_min, dist_max, dist_aver, dist_std

def calEachPointError(plane_func_1x4, points_nx3):
    signa = plane_func_1x4[3] * (-1)
    normal_vec = np.array(plane_func_1x4[:3]).T
    points_nx3 = np.array(points_nx3)
    error_item = signa - np.dot(points_nx3, normal_vec)
    # print("error item: ", error_item)
    # print("error_item: ", error_item)
    return error_item

def calAverageError(plane_func_1x4, points_nx3):
    # signa = plane_func_1x4[3]
    # normal_vec = np.array(plane_func_1x4[:3]).T
    # points_nx3 = np.array(points_nx3)
    # EAVG = np.sum(signa - np.dot(points_nx3, normal_vec)) / np.linalg.norm(points_nx3)
    error_item = calEachPointError(plane_func_1x4, points_nx3)
    EAVG = np.abs(np.sum(error_item)) / np.linalg.norm(points_nx3)
    print("EAVG: ", EAVG)
    return EAVG

def calAverageRMSE(plane_func_1x4, points_nx3):
    error_item = calEachPointError(plane_func_1x4, points_nx3)
    RMSE = np.sqrt(np.sum(np.power(error_item, 2)) / np.linalg.norm(points_nx3))
    print("RMSE: ", RMSE)
    return RMSE

def calOrthogonalityMetric(plane_func_1x4_1, plane_func_1x4_2):
    n_3x1 = plane_func_1x4_1[:3]
    n_3x2 = plane_func_1x4_2[:3]
    cosin = np.dot(n_3x1, n_3x2) / (np.linalg.norm(n_3x1) * np.linalg.norm(n_3x2))
    angle = np.arccos(cosin)
    angle = np.rad2deg(angle)
    print("angle: ", angle)
    return angle


def main():
    import random
    import open3d
    from plane.bin.tool import cvtNumpyToPCD, cvtPCDToNumpy
    # point_cloud = np.loadtxt("../res/input/plane1.txt")

    point_cloud = np.loadtxt("/home/pi/Music/my_test_project/open3d/points_ahead.txt")
    pcd = cvtNumpyToPCD(point_cloud)
    print(len(point_cloud))
    pcd_sample = open3d.voxel_down_sample(pcd, voxel_size=0.25)
    point_cloud_sample = cvtPCDToNumpy(pcd_sample)
    print(len(point_cloud_sample))
    plane_f = planeFitBySVD(point_cloud_sample)
    dist, dist_min, dist_max, dist_mean, dist_std = planeEvaluate(point_cloud_sample, plane_f)
    print("dist_min: ", dist_min)
    print("dist_max: ", dist_max)
    print("dist_mean: ", dist_mean)
    print("dist_std: ", dist_std)
    calAverageError(plane_f, point_cloud_sample)
    calAverageRMSE(plane_f, point_cloud_sample)

def angleTest():
    # fr = np.array([-0.5647338958421042, 0.4347623999508783, 0.7014679482884104, -138.675674955067])
    # fl = np.array([-0.6776125709993795, -0.37077625371903833, -0.6351111503521933, 150.0181195436238])
    # fa = np.array([0.11616488015568698, -0.49023765961029103, 0.8638129182399619, -151.28133041319973])

    fr = np.array([-0.564465, +0.434542, 0.701676, 0])
    fl = np.array([-0.677673, -0.370952, -0.634944, 0])
    fa = np.array([0.116142, -0.490388, 0.863373, 0])


    calOrthogonalityMetric(fr, fl)
    calOrthogonalityMetric(fr, fa)
    calOrthogonalityMetric(fl, fa)

if __name__ == "__main__":
    main()
    # angleTest()
