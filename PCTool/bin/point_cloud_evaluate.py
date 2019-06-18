#!/usr/bin/python3
# -*- coding: utf-8 -*-
__Author__ = 'csd'
__version__ = "1.0"
__data__ = "2019.05.17"

import os
import sys
import numpy as np
from PCTool.bin.svd_fit_plane import SVDFitPlane
from PCTool.bin import pointcloudtool as pcl

class PointCloudEvaluate():
    def __init__(self, src_points, crop_volume_file, dst_points, width_mm=15.5, height_mm=40.0):
        """
        :param src_points: 原始点云数据,nparray_nx3,参考"../res/sample_point_cloud.png"
        :param crop_volume_file: 点云切割json配置文件,list,由pcl.cropGeometryGenerator()生成
        :param dst_points: 目标点云数据,nparray_nx3
        :param width_mm: 待评估点云平面宽度
        :param height_mm: 待评估点云平面长度
        """
        self._src_points = pcl.cvtNumpyToPCD(src_points)
        self._src_points_nx3 = src_points
        self._crop_vol_file = crop_volume_file
        self._plane_width = width_mm
        self._plane_height = height_mm
        self._dst_points = pcl.cvtNumpyToPCD(dst_points)
        self._dst_points_nx3 = dst_points

        self._fitter = SVDFitPlane()
        self._matcher = pcl.ICPMatch(self._src_points, self._dst_points, showResult=False)

    def _pointsTransform(self):
        '''
        目标点云经过配准后,变换到原始点云所在的位置,在用原始点云生成的切割配置文件分割点云
        :return:
        '''
        T_rac, T_icp = self._matcher.globalRansacMatchAndICPRefine()
        Transfrom_matrix_target_to_source = np.linalg.inv(T_icp.transformation)
        rotation_target_to_source = Transfrom_matrix_target_to_source[0:3, 0:3]
        transform_target_to_source = Transfrom_matrix_target_to_source[0:3, -1].reshape(3, 1)
        target_points_3xn = self._dst_points_nx3.T
        target_points_3xn_new = np.dot(rotation_target_to_source, target_points_3xn) + transform_target_to_source
        self._dst_points_nx3 = target_points_3xn_new.T
        self._dst_points = pcl.cvtNumpyToPCD(self._dst_points_nx3)


    def _pointsPreprocess(self):
        '''
        点云切割,并进行适当下采样,减少点云数据量,这边会切出5个
        :return:
        '''
        pcd_crop_list = pcl.cropPointByVolume(self._dst_points, self._crop_vol_file)
        self._single_plane_pcd = pcd_crop_list[0]
        self._multi_plane_pcd = pcd_crop_list[1:]
        down_sample = pcl.pointsDownSample(self._single_plane_pcd, 8000, 0.05)
        self._single_plane_pcd = down_sample
        self._single_plane_np = pcl.cvtPCDToNumpy(self._single_plane_pcd)
        self._multi_plane_np = []
        for i in range(0, len(self._multi_plane_pcd)):
            down_sample = pcl.pointsDownSample(self._multi_plane_pcd[i], 8000, 0.05)
            self._multi_plane_pcd[i] = down_sample
            self._multi_plane_np.append(pcl.cvtPCDToNumpy(down_sample))

    def fitPlane(self):
        '''
        拟合平面
        :return:
        '''
        self._single_plane_func = self._fitter.fit(self._single_plane_np)
        self._multi_plane_func = []
        for point_data in self._multi_plane_np:
            self._multi_plane_func.append(self._fitter.fit(point_data))

    def _calPointsToPlaneDistance(self, planeFunc, points_nx3):
        '''
        计算点云到拟合平面的距离
        :param planeFunc: 目标点云拟合方程
        :param points_nx3: 目标点云数据
        :return: dictionary,包括最大值,最小值,标准差,均值
        '''
        dist = []
        normal_vec = planeFunc[:3]
        intercept = planeFunc[3]
        sqrt_val = np.sqrt(normal_vec[0] * normal_vec[0] + normal_vec[1] * normal_vec[1] +
                           normal_vec[2] * normal_vec[2])
        for point in points_nx3:
            dist_item = abs((normal_vec[0] * point[0] + normal_vec[1] * point[1] + normal_vec[2] * point[2] + intercept) / sqrt_val)
            dist.append(dist_item)
        dist_min = min(dist)
        dist_max = max(dist)
        dist_aver = np.mean(dist)
        dist_std = np.std(dist)
        dist = {}
        dist["min"] = dist_min
        dist["max"] = dist_max
        dist["aver"] = dist_aver
        dist["std"] = dist_std
        return dist

    def _calEachPointError(self, planeFunc, points_nx3):
        signa = planeFunc[3] * (-1)
        normal_vec = np.array(planeFunc[:3]).T
        # normal_vec = planeFunc[:3].T

        error_item = signa - np.dot(points_nx3, normal_vec)
        return error_item

    def _calAverageError(self, planeFunc, points_nx3):
        '''
        计算平均误差
        :param planeFunc: 平面方程
        :param points_nx3: 点云数据
        :return: 平均误差
        '''
        error_item = self._calEachPointError(planeFunc, points_nx3)
        EAVG = np.abs(np.sum(error_item)) / np.linalg.norm(points_nx3)
        # print("EAVG: ", EAVG)
        return EAVG

    def _calAverageRMSE(self, planeFunc, points_nx3):
        '''
        RMSE 均方跟误差
        :param planeFunc: 平面方程
        :param points_nx3: 点云数据
        :return: RMSE
        '''
        error_item = self._calEachPointError(planeFunc, points_nx3)
        RMSE = np.sqrt(np.sum(np.power(error_item, 2)) / np.linalg.norm(points_nx3))
        # print("RMSE: ", RMSE)
        return RMSE

    def calDistError(self, planeFunc, points_nx3):
        res_dist = self._calPointsToPlaneDistance(planeFunc, points_nx3)
        eavg = self._calAverageError(planeFunc, points_nx3)
        res_dist["EAVG"] = eavg
        rmse = self._calAverageRMSE(planeFunc, points_nx3)
        res_dist["RMSE"] = rmse
        return res_dist

    def calOrthogonalityMetric(self, plane_func_1x4_1, plane_func_1x4_2):
        '''
        计算平面之间的夹角(正交性)
        :param plane_func_1x4_1: 平面1方程
        :param plane_func_1x4_2: 平面2方程
        :return: 平面夹角(deg)
        '''
        n_3x1 = plane_func_1x4_1[:3]
        n_3x2 = plane_func_1x4_2[:3]
        cosin = np.dot(n_3x1, n_3x2) / (np.linalg.norm(n_3x1) * np.linalg.norm(n_3x2))
        angle = np.arccos(cosin)
        angle = np.rad2deg(angle)
        # print("angle: ", angle)
        return angle

    def calAccuracy(self):
        '''
        计算点云精度
        :return: list [点云的水平距离,竖直距离,水平距离与实际距离之差,竖直距离与实际距离只差]
        '''
        points_nx3 = pcl.cvtPCDToNumpy(self._single_plane_pcd)
        x_distance = abs(max(points_nx3[:, 1]) - min(points_nx3[:, 1]))
        y_distance = abs(max(points_nx3[:, 0]) - min(points_nx3[:, 0]))
        accuracy = [x_distance, y_distance, abs(x_distance - self._plane_width), abs(y_distance - self._plane_height)]
        return accuracy


    def calSampleRate(self):
        '''
        计算采样率
        :return: pix/mm2
        '''
        points_nx3 = pcl.cvtPCDToNumpy(self._single_plane_pcd)
        sample_rate = len(points_nx3) / self._plane_width * self._plane_height
        return sample_rate

    def runEvaluate(self):
        res = {
        }
        self._pointsTransform()
        self._pointsPreprocess()
        self.fitPlane()
        # 采样率
        sample_rate = self.calSampleRate()
        res["sample_rate"] = sample_rate
        # 采样精度
        accuracy = self.calAccuracy()
        res["accuracy"] = accuracy
        # 平面上的点和自身拟合平面的评估值
        dist_res = self.calDistError(self._multi_plane_func[0], self._multi_plane_np[0])
        res["plane1_eva"] = dist_res
        dist_res = self.calDistError(self._multi_plane_func[1], self._multi_plane_np[1])
        res["plane2_eva"] = dist_res
        dist_res = self.calDistError(self._multi_plane_func[2], self._multi_plane_np[2])
        res["plane3_eva"] = dist_res
        dist_res = self.calDistError(self._multi_plane_func[3], self._multi_plane_np[3])
        res["plane4_eva"] = dist_res
        # 平面相对距离
        dist_res = self.calDistError(self._multi_plane_func[0], self._multi_plane_np[2])
        res["relative_dsit_1x3"] = dist_res
        dist_res = self.calDistError(self._multi_plane_func[1], self._multi_plane_np[3])
        res["relative_dsit_2x4"] = dist_res
        # 90度角
        angle = self.calOrthogonalityMetric(self._multi_plane_func[0], self._multi_plane_func[1])
        res["angle_1x2"] = angle
        angle = self.calOrthogonalityMetric(self._multi_plane_func[1], self._multi_plane_func[2])
        res["angle_2x3"] = angle
        angle = self.calOrthogonalityMetric(self._multi_plane_func[2], self._multi_plane_func[3])
        res["angle_3x4"] = angle
        # 平行平面
        angle = self.calOrthogonalityMetric(self._multi_plane_func[0], self._multi_plane_func[2])
        res["angle_1x3"] = angle
        angle = self.calOrthogonalityMetric(self._multi_plane_func[1], self._multi_plane_func[3])
        res["angle_2x4"] = angle
        return res

def main():
    pass


if __name__ == "__main__":
    main()
