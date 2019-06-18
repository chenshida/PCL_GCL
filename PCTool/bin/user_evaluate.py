#!/usr/bin/python3
# -*- coding: utf-8 -*-
__Author__ = 'csd'
__version__ = "1.0"
__data__ = "2019.05.17"

import os
import sys
from PCTool.bin import pointcloudtool as pcl
from PCTool.bin.svd_fit_plane import SVDFitPlane
from PCTool.bin.point_cloud_evaluate import pointCloudEvaluate


def main():
    # file_name = "/home/pi/Music/my_test_project/open3d/points.pcd"
    # vol_file = ["../res/right.json",
    #             "../res/left.json",
    #             "../res/up.json"]

    file_name = "/home/pi/Desktop/cali_struct_light/6-5/points.txt"
    vol_file = ["./p5.json",
                "./p6.json",
                "./p7.json",
                "./p8.json",
                ]
    points_src = pcl.readPoints(file_name)
    # points_src[87563] = [0, 4.35, 255.61]
    point_pcd = pcl.cvtNumpyToPCD(points_src)
    fit = SVDFitPlane()
    evaluater = pointCloudEvaluate()
    points_cut_item = pcl.cropPointByVolume(point_pcd, vol_file)
    plane_func_nx4 = []
    points_item = []
    for i in range(0, len(points_cut_item)):
        points_dst = pcl.pointsDownSample(points_cut_item[i], expectPoints=8000, step=0.05)
        # pcl.showPointCloud(points_dst)
    # print(len(points_dst))
    # showPointCloud(points_dst)
        point_down_sample = pcl.cvtPCDToNumpy(points_dst)
        plane_func = fit.fit(point_down_sample)
        print(plane_func)
        plane_func_nx4.append(plane_func)
        points_item.append(pcl.cvtPCDToNumpy(points_cut_item[i]))
        evaluater.runEvaluate(point_down_sample, plane_func)
        res = evaluater.evaluate
        for key, val in res.items():
            print((key, val))
    angle1_90 = evaluater.calOrthogonalityMetric(plane_func_nx4[0], plane_func_nx4[1])
    angle2_90 = evaluater.calOrthogonalityMetric(plane_func_nx4[1], plane_func_nx4[2])
    angle3_90 = evaluater.calOrthogonalityMetric(plane_func_nx4[2], plane_func_nx4[3])

    print("angle1_90: ", angle1_90)
    print("angle2_90: ", angle2_90)
    print("angle3_90: ", angle3_90)
    print("\n")
    evaluater.runEvaluate(points_item[0], plane_func_nx4[2])
    res = evaluater.evaluate
    for key, val in res.items():
        print((key, val))

    print("\n")
    evaluater.runEvaluate(points_item[1], plane_func_nx4[3])
    res = evaluater.evaluate
    for key, val in res.items():
        print((key, val))

    # angle1 = evaluater.calOrthogonalityMetric(plane_func_nx4[0], plane_func_nx4[1])
    # angle2 = evaluater.calOrthogonalityMetric(plane_func_nx4[2], plane_func_nx4[3])
    angle3_0 = evaluater.calOrthogonalityMetric(plane_func_nx4[0], plane_func_nx4[2])
    angle4_0 = evaluater.calOrthogonalityMetric(plane_func_nx4[1], plane_func_nx4[3])
    # print("angle1: ", angle1)
    # print("angle2: ", angle2)
    print("angle3_0: ", angle3_0)
    print("angle4_0: ", angle4_0)

def main2():
    from PCTool.bin.point_cloud_evaluate import PointCloudEvaluate
    source_file_name = "/home/pi/Desktop/cali_struct_light/6-18/source/match_source.ply"
    vol_file = ["/home/pi/Desktop/cali_struct_light/6-18/source/plane.json",
                "/home/pi/Desktop/cali_struct_light/6-18/source/p1.json",
                "/home/pi/Desktop/cali_struct_light/6-18/source/p2.json",
                "/home/pi/Desktop/cali_struct_light/6-18/source/p3.json",
                "/home/pi/Desktop/cali_struct_light/6-18/source/p4.json",
                ]
    source_points = pcl.readPoints(source_file_name)
    target_file_name = "/home/pi/Desktop/cali_struct_light/6-18/target/points_target.txt"
    target_points = pcl.readPoints(target_file_name)
    target_points_cut = pcl.pointCloudCut(target_points, lower_mm=-375, upper_mm=-340)
    hh = PointCloudEvaluate(source_points, vol_file, target_points_cut)
    res = hh.runEvaluate()
    for key, data in res.items():
        print(key, data)
    # print("res: \n", res)

def json_test():
    import json
    import numpy as np
    from toolbox import vgl
    file_name = "/home/pi/Desktop/cali_struct_light/6-18/source/plane.json"
    with open(file_name, 'r') as load_f:
        load_dict = json.load(load_f)

    bounding_polygon = load_dict["bounding_polygon"]
    print("bounding_polygon: ", bounding_polygon)
    # print("bounding_polygon length: ", len(bounding_polygon))
    T = np.array([[-1.03526900e-02,  9.99281112e-01, -3.64702819e-02, -2.59859643e+01],
                  [-9.97525943e-01, -1.28569131e-02, -6.91136292e-02, -1.72699261e+01],
                  [-6.95328395e-02,  3.56645403e-02,  9.96941937e-01, -1.84413349e+00],
                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    pose = vgl.T2Pose(T)
    print("pose: \n", pose)
    R = T[:3, :3]
    t = T[:3, 3]
    print("R: \n", R)
    print("t: \n", t)
    # bounding_polygon_rotation = []
    # for pts in bounding_polygon:
    #     pts = np.array(pts).T
    #     print("pts: ", pts)
    #     new_pts = (np.dot(R, pts) + t).T
    #     # new_pts[1] = new_pts[0]
    #     new_pts[1] = 0
    #     bounding_polygon_rotation.append(new_pts)
    # print("bounding_polygon_rotation: ", np.array(bounding_polygon_rotation).reshape(-1, 3))
    #
    # x_max = np.array([load_dict["axis_max"], 0, 0])
    # x_min = np.array([load_dict["axis_min"], 0, 0])
    # x_max_new = (np.dot(R, x_max) + t).T
    # x_min_new = (np.dot(R, x_min) + t).T
    #
    # # print("x_max_new: ", x_max_new)
    # # exit()
    #
    # new_json = {}
    # new_json["axis_max"] = load_dict["axis_max"]
    # new_json["axis_min"] = load_dict["axis_min"]
    # new_json["class_name"] = load_dict["class_name"]
    # new_json["orthogonal_axis"] = load_dict["orthogonal_axis"]
    # # new_json["orthogonal_axis"] = "Y"
    # new_json["version_major"] = load_dict["version_major"]
    # new_json["version_minor"] = load_dict["version_minor"]
    # new_json["bounding_polygon"] = load_dict["bounding_polygon"]
    #
    # with open("new_plane.json", 'w') as dump_f:
    #     json.dump(new_json, dump_f)
    # # with open("new_plane.json", 'r') as load_f:
    # #     new_json_loder = json.loads(load_f)
    # # new_json_loder = json.dumps(new_json)
    # new_json_file = [
    #     "./new_plane.json",
    # ]

    target_points_path = "/home/pi/Desktop/cali_struct_light/6-18/target/points_target.txt"
    points_target = pcl.readPoints(target_points_path)
    points_3xn = points_target.T
    T_INV = np.linalg.inv(T)
    R1, t1 = vgl.T2Rt(T_INV)
    points_new = np.dot(R1, points_3xn) + t1
    points_new = points_new.T
    pcl.showPointCloud([points_new])



    points_target_pcd = pcl.cvtNumpyToPCD(points_new)
    points_crop = pcl.cropPointByVolume(points_target_pcd, [file_name])
    pcl.showPointCloud(points_crop)


    # json.loads(file_name)

if __name__ == "__main__":
    # main()
    main2()
    # json_test()