#!/usr/bin/python3
# -*- coding: utf-8 -*-
__Author__ = 'csd'
__version__ = "1.0"
__data__ = "2019.05.15"

import os
import sys
import open3d
import numpy as np

def cvtNumpyToPCD(ndarray_nx3):
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(ndarray_nx3)
    return pcd

def showPointCloud(point_cloud):
    """
    show pointcloud by py3d(open3d)
    """
    if isinstance(point_cloud, open3d.PointCloud):
        open3d.draw_geometries([point_cloud])
    elif isinstance(point_cloud, np.ndarray):
        pcd = cvtNumpyToPCD(point_cloud)
        open3d.draw_geometries([pcd])
    else:
        raise ValueError("object {} is not a numpy array or point cloud" % point_cloud)

def point_cloud_cut(src_pc, lower_mm, upper_mm, axis=2):
    point_cloud_nx3 = src_pc[src_pc[:, axis] > lower_mm]
    point_cloud_nx3 = point_cloud_nx3[point_cloud_nx3[:, axis] < upper_mm]
    # point_cloud_nx3 = point_cloud_nx3[point_cloud_nx3[:, 1] > y_lower_mm]
    # point_cloud_nx3 = point_cloud_nx3[point_cloud_nx3[:, 1] < y_upper_mm]
    # point_cloud_nx3 = point_cloud_nx3[point_cloud_nx3[:, 0] > x_lower_mm]
    # point_cloud_nx3 = point_cloud_nx3[point_cloud_nx3[:, 0] < x_upper_mm]
    return point_cloud_nx3

def savePCD(filename, point_cloud):
    path, _ = os.path.split(filename)
    if not os.path.exists(path):
        os.mkdir(path)
    if isinstance(point_cloud, np.ndarray):
        pcd = cvtNumpyToPCD(point_cloud)
        open3d.write_point_cloud(filename, pcd)
    elif isinstance(point_cloud, open3d.PointCloud):
        open3d.write_point_cloud(filename, point_cloud)
    else:
        raise ValueError("object {} is not a numpy array or point cloud" % point_cloud)

def cvtPCDToNumpy(pcd_load):
    xyz_load = np.asarray(pcd_load.points)
    return xyz_load

def readPCD(filename):
    pcd = open3d.read_point_cloud(filename)
    return pcd

def main():
    points = np.loadtxt("points.txt")
    pcd = cvtNumpyToPCD(points)
    showPointCloud(pcd)
    savePCD("./points.pcd", pcd)

def main1(pcdFilename, txtFilename):
    pcd = readPCD(pcdFilename)
    points = cvtPCDToNumpy(pcd)
    np.savetxt(txtFilename, points)

if __name__ == "__main__":
    # main()
    main1("project_l.pcd", './points_left.txt')
    main1("project_r.pcd", './points_right.txt')
    main1("project_ahead.pcd", './points_ahead.txt')
