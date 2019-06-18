#!/usr/bin/python3
# -*- coding: utf-8 -*-
__Author__ = 'csd'
__version__ = "1.0"
__data__ = "2019.05.17"

import os
import sys
import numpy as np
import open3d
import threading
import time
import copy


def cvtNumpyToPCD(ndarray_nx3):
    '''
    convert np.array data type to open3d-pcd type
    :param ndarray_nx3: nx3 point cloud of ndarray
    :return: pcd for open3d
    '''
    if isinstance(ndarray_nx3, open3d.PointCloud):
        return ndarray_nx3
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(ndarray_nx3)
    return pcd

def pointCloudCut(src_pc, lower_mm, upper_mm, axis=2):
    point_cloud_nx3 = src_pc[src_pc[:, axis] > lower_mm]
    point_cloud_nx3 = point_cloud_nx3[point_cloud_nx3[:, axis] < upper_mm]
    return point_cloud_nx3

def cvtPCDToNumpy(pcd):
    '''
    convert pcd to numpy
    :param pcd: PCD data
    :return: np.ndarray Nx3
    '''
    return np.asarray(pcd.points)

def readPoints(filename):
    '''
    read point cloud from file, file type maybe pcd,ply,txt
    :param filename: string of point cloud locate path
    :return: ndarray of point cloud nx3
    '''
    assert isinstance(filename, str)
    if '.pcd' in filename or '.ply' in filename:
        pcd = open3d.read_point_cloud(filename)
        points = np.asarray(pcd.points)
    elif 'txt' in filename:
        points = np.loadtxt(filename)
    else:
        raise ValueError("invalid point cloud file type, it must txt, pcd or ply")
    return points

def getPointCapability(pcd):
    points_nx3 = np.asarray(pcd.points)
    cap = {}
    cap["point_size"] = len(points_nx3)
    cap["x_min"] = np.min(points_nx3[:, 0])
    cap["x_max"] = np.max(points_nx3[:, 0])
    cap["y_min"] = np.min(points_nx3[:, 1])
    cap["y_max"] = np.max(points_nx3[:, 1])
    cap["z_min"] = np.min(points_nx3[:, 2])
    cap["z_max"] = np.max(points_nx3[:, 2])
    cap["x_size"] = cap["x_max"] - cap["x_min"]
    cap["y_size"] = cap["y_max"] - cap["y_min"]
    cap["z_size"] = cap["z_max"] - cap["z_min"]
    return cap

def cropPointByVolume(pcd, volumeFileList):
    '''
    crop a point cloud by volume
    :param pcd: point cloud with open3d.PointCloud object
    :param volumeFileList: a list of json file which will use for crop point cloud
    :return: croped point cloud
    '''
    points_crop_list = []
    for volumeFile in volumeFileList:
        if isinstance(volumeFile, str):
            volume = open3d.read_selection_polygon_volume(volumeFile)
        else:
            volume = volumeFile
        points_crop = volume.crop_point_cloud(pcd)
        # points_np = np.asarray(points_crop.points)
        points_crop_list.append(points_crop)
    return points_crop_list

def cropGeometryGenerator(pcd):
    '''
    create crop json file
    :param pcd: source point cloud from file or memory
    :return: a list crop json file
    '''
    print("Demo for manual geometry cropping")
    print("1) Press 'X' twice to align geometry with negative direction of x-axis")
    print("   or Press 'Y' twice to align geometry with negative direction of y-axis")
    print("   or Press 'z' twice to align geometry with negative direction of z-axis")
    print("   this step must done, otherwise, the crop volume can not generate")
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    open3d.draw_geometries_with_editing([pcd])

def showPointCloud(point_cloud_list):
    """
    show pointcloud by py3d(open3d)
    it will block current thread
    """
    assert isinstance(point_cloud_list, list)
    pcd_list = []
    for point_cloud in point_cloud_list:
        if isinstance(point_cloud, open3d.PointCloud):
            pcd_list.append(point_cloud)
        elif isinstance(point_cloud, np.ndarray):
            pcd = cvtNumpyToPCD(point_cloud)
            pcd_list.append(pcd)
        else:
            raise ValueError("object {} is not a numpy array or point cloud" % point_cloud)
    open3d.draw_geometries(pcd_list)
    # if isinstance(point_cloud, open3d.PointCloud):
    #     open3d.draw_geometries([point_cloud])
    # elif isinstance(point_cloud, np.ndarray):
    #     pcd = cvtNumpyToPCD(point_cloud)
    #     open3d.draw_geometries([pcd])
    # else:
    #     raise ValueError("object {} is not a numpy array or point cloud" % point_cloud)

def pointsDownSample(pcd, expectPoints=10000, step=0.1):
    '''
    down sample point cloud
    :param pcd: points data of open3d.PointCloud
    :param expectPoints: after dowmsample, whole points must less than the value
    :param step: each step increase of the radium
    :return: after down sample data
    '''
    points_num = len(pcd.points)
    if points_num < expectPoints:
        return pcd
    voxel_size = 0.1

    while points_num > expectPoints:
        pcd_sample = open3d.voxel_down_sample(pcd, voxel_size=voxel_size)
        points_num = len(pcd_sample.points)
        voxel_size += step
    return pcd_sample

def pick_points(pcd):
    """
    pick points from a showing pcd
    :param pcd: input data, it must convert to open3d.PointCloud data type
    :return: the selected points index and it`s value
    """
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = open3d.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run() # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

class ICPMatch():
    def __init__(self, sourceRaw, targetRaw, voxelSize_mm=1.5,
                 autoThreshold=True, distanceThresholdRansac_mm=7.5, distanceThresholdICP_mm=2, showResult=True):
        self._source = cvtNumpyToPCD(sourceRaw)
        self._target = cvtNumpyToPCD(targetRaw)
        self._voxel_size_m = voxelSize_mm # open3d data type
        # print("self._voxel_size_m: ", self._voxel_size_m)
        # exit()
        self._show_result = showResult
        if autoThreshold:
            self._distance_threshold_ransac_m = self._voxel_size_m * 1.5
            self._distance_threshold_icp_m = self._voxel_size_m * 0.4
        else:
            self._distance_threshold_ransac_m = distanceThresholdRansac_mm
            self._distance_threshold_icp_m = distanceThresholdICP_mm

    def _preprocessPointCloud(self, pcd, voxelSize):
        print("Downsample with a voxel size %.3f." % voxelSize)
        # pcd_down = pcd.voxel_down_sample(voxelSize)
        pcd_down = open3d.voxel_down_sample(pcd, voxelSize)
        # cl, pcd_out_remove = open3d.statistical_outlier_removal(pcd_down, nb_neighbors=20, std_ratio=2.0)

        radius_normal = voxelSize * 2
        print("Estimate normal with search radius %.3f." % radius_normal)
        # pcd_down.estimate_normals(open3d.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        open3d.estimate_normals(pcd_down, open3d.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxelSize * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = open3d.compute_fpfh_feature(
                pcd_down,open3d.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

        return pcd_down, pcd_fpfh

    def prepareDataset(self):
        # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
        #                      [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        # self._source.transform(trans_init)
        source_down, source_fpfh = self._preprocessPointCloud(self._source, self._voxel_size_m)
        target_down, target_fpfh = self._preprocessPointCloud(self._target, self._voxel_size_m)
        return source_down, source_fpfh, target_down, target_fpfh

    def showMatchResult(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        open3d.draw_geometries([source_temp, target_temp])

    def globalRansacMatchAndICPRefine(self):
        source_down, source_fpfh, target_down, target_fpfh = self.prepareDataset()

        # showPointCloud([target_down])
        # exit()
        # result_ransac = open3d.registration_ransac_based_on_feature_matching(
        #         source_down, target_down, source_fpfh, target_fpfh, self._distance_threshold_ransac_m,
        #         open3d.TransformationEstimationPointToPoint(False), 4, [
        #         open3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        #         open3d.CorrespondenceCheckerBasedOnDistance(
        #         self._distance_threshold_ransac_m)
        # ], open3d.RANSACConvergenceCriteria(4000000, 500))
        result_ransac = open3d.registration_fast_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh,
                open3d.FastGlobalRegistrationOption(
                maximum_correspondence_distance=self._distance_threshold_ransac_m)
         )
        # ], open3d.RANSACConvergenceCriteria(40000, 500))
        if self._show_result:
            self.showMatchResult(source_down, target_down, result_ransac.transformation)
        # result_icp = open3d.registration_icp(
        #         self._source, self._target, self._distance_threshold_icp_m, result_ransac.transformation,
        #         open3d.TransformationEstimationPointToPlane())
        result_icp = open3d.registration_icp(
                source_down, target_down, self._distance_threshold_icp_m, result_ransac.transformation,
                open3d.TransformationEstimationPointToPlane())
        if self._show_result:
            self.showMatchResult(self._source, self._target, result_icp.transformation)
        return result_ransac, result_icp





# TODO has bug, wait to fix
class NonBlockVisualization():
    """
    non block for showing point cloud

    Example
    ---------
    >>> file_name = "xxx/xxx.pcd"
    >>> file_name2 = "xxx/xxx.pcd"
    >>> point = readPoints(file_name)
    >>> pcd = cvtNumpyToPCD(point)
    >>> window = NonBlockVisualization()
    >>> window.addGeometry(pcd)
    >>> window.updateViewer()

    >>> print("new point cloud")
    >>> point = readPoints(file_name2)
    >>> window.updatePoints(point)
    >>> window.updateViewer()
    """
    def __init__(self, windowName='Open3D', width=1920, height=1080, left=50, right=50):
        self._pcd = open3d.PointCloud()
        self._thread = None
        self._vis = open3d.Visualizer()
        self._vis.create_window(windowName, width, height, left, right)

    def addGeometry(self, source_pcd):
        self._pcd = source_pcd
        self._vis.add_geometry(self._pcd)

    def showPointCloud(self):
        # if self._thread is not None:
        #     if self._thread.isAlive():
        #         self._thread.join()
        #     self._thread = None

        self._thread = threading.Thread(target=self.updateViewerThread)
        self._thread.setDaemon(True)

        self._thread.start()
        # self._thread.join()


    def updateViewerThread(self):
        while True:
            self.updateViewer()
            time.sleep(0.01)

    def updateViewer(self):
        self._vis.update_geometry()
        self._vis.poll_events()
        self._vis.update_renderer()

    def updatePoints(self, ndarray_nx3):
        self._pcd.points = open3d.Vector3dVector(ndarray_nx3)

    def destroyWindow(self):
        self._vis.destroy_window()

    # def removeGeometry(self, source_pcd):
    #     self._vis.remove_geometry(source_pcd)



def cut():
    # file_name = "/home/pi/Music/my_test_project/open3d/project.pcd"
    # file_name = "/home/pi/Desktop/cali_struct_light/6-18/source/points_source.txt"
    file_name = "/home/pi/Desktop/cali_struct_light/6-18/target/points_target.txt"
    # file_name = "/home/pi/Music/my_test_project/open3d/points_left.txt"
    # file_name = "/home/pi/Music/my_test_project/open3d/cropped.ply"
    # vol_file = ["/home/pi/PycharmProjects/MyProject/PCTool/bin/z_alies_right.json",
    #             "/home/pi/PycharmProjects/MyProject/PCTool/bin/z_alies_left.json",
    #             "/home/pi/PycharmProjects/MyProject/PCTool/bin/z_alies_up.json"]
    point = readPoints(file_name)
    # point[87563] = [0, 4.35, 255.61]
    pcd = cvtNumpyToPCD(point)
    cropGeometryGenerator(pcd)
    # points_cut_item = cropPointByVolume(file_name, vol_file)
    # for i in range(0, len(points_cut_item)):
    #     showPointCloud(points_cut_item[i])

def createPoints():
    X = np.arange(-5, 5, 0.025)
    Y = np.arange(-5, 5, 0.025)
    X,Y = np.meshgrid(X,Y)
    R = np.sqrt(X**2+Y**2)
    Z = np.sin(R)

    X1 = X.reshape(-1)
    Y1 = Y.reshape(-1)
    Z1 = Z.reshape(-1)
    # XYZ = np.vstack([X1, Y1, Z1]).T
    XYZ = np.vstack([Z1, Y1, X1]).T
    return XYZ

def createPoints2():
    X = np.arange(-5, 5, 0.025)
    Y = np.arange(-5, 5, 0.025)
    X,Y = np.meshgrid(X,Y)
    # Z = np.sqrt(X**2+Y**2)
    # Z = 2+np.cos(Z)+np.sin(Y)
    Z = np.cos(X) + np.sin(X)
    # X = np.cos(X) + 2
    # Y = np.sin(Y) + 2

    X1 = X.reshape(-1)
    Y1 = Y.reshape(-1)
    Z1 = Z.reshape(-1)
    XYZ = np.vstack([X1, Y1, Z1]).T
    # XYZ = np.vstack([Z1, Y1, X1])q.T
    return XYZ

def createPoints3():
    X = np.arange(-5, 5, 0.025)
    Y = np.arange(-5, 5, 0.025)
    X,Y = np.meshgrid(X,Y)
    Z = np.sqrt(X**2+Y**2)
    Z = 2+np.cos(Z)
    # Z = np.cos(X) + np.sin(X)
    # X = np.cos(X) + 2
    # Y = np.sin(Y) + 2

    X1 = X.reshape(-1)
    Y1 = Y.reshape(-1)
    Z1 = Z.reshape(-1)
    XYZ = np.vstack([X1, Y1, Z1]).T
    # XYZ = np.vstack([Z1, Y1, X1])q.T
    return XYZ

def visTest():
    import time


    point = createPoints3()
    pcd = cvtNumpyToPCD(point)
    # showPointCloud(pcd)
    window = NonBlockVisualization()
    window.addGeometry(pcd)
    window.updateViewer()
    window.showPointCloud()
    time.sleep(5)
    print("new point cloud")
    point = createPoints()
    # point = readPoints(file_name2)
    window.updatePoints(point)
    for i in range(0, 100):

        window.updateViewer()
    # time.sleep(5)

def pickPointsTest():
    file_name = "/home/pi/PycharmProjects/MyProject/PCTool/res/cali_points/distance/points.txt"
    point = readPoints(file_name)
    pcd = cvtNumpyToPCD(point)
    pick_points(pcd)

def getPointCloudCapTest():
    file_name = "/home/pi/PycharmProjects/MyProject/PCTool/res/cali_points/demsity/38x40.ply"
    points = readPoints(file_name)
    point_pcd = cvtNumpyToPCD(points)
    cap = getPointCapability(point_pcd)
    for key, val in cap.items():
        print(key, val)

if __name__ == "__main__":
    cut()
    exit()
    # getPointCloudCapTest()
    # visTest()
    # createPoints()
    # createPoints2()
    # pickPointsTest()
    src_path = "/home/pi/Desktop/cali_struct_light/6-18/source/match_source.ply"
    dst_path = "/home/pi/Desktop/cali_struct_light/6-18/target/points_target.txt"
    # src_path = "/home/pi/Documents/Open3D/examples/TestData/ICP/cloud_bin_0.pcd"
    # dst_path = "/home/pi/Documents/Open3D/examples/TestData/ICP/cloud_bin_1.pcd"
    src_np = readPoints(src_path)
    dst_np = readPoints(dst_path)
    dst_cut = pointCloudCut(dst_np, lower_mm=-375, upper_mm=-340)
    matcher = ICPMatch(src_np, dst_cut, voxelSize_mm=1.5)

    # src = cvtNumpyToPCD(src_np)
    # dst = cvtNumpyToPCD(dst_np)
    T_rac, T_icp = matcher.globalRansacMatchAndICPRefine()
    print("T Ransac: \n", T_rac.transformation)
    print("T ICP: \n", T_icp.transformation)
