#!/usr/bin/python3
# -*- coding: utf-8 -*-
__Author__ = 'csd'
__version__ = "1.0"
__data__ = "2019.05.16"

import os
import sys
import numpy as np
import scipy

def random_partition(n, n_data):
    all_index = np.arange(n_data)
    np.random.shuffle(all_index)
    idexs1 = all_index[:n]
    idexs2 = all_index[n:]
    return idexs1, idexs2

def ransac(data, model, minSample=-1, maxIteration=-1, threshold=0.01, minFitModelThreshold=-1, debug=False, returnAll=False):
    '''

    :param data: sample data
    :param model: user define function model
    :param minSample: min sample to fit model select from sample data
    :param maxIteration: max iteration
    :param threshold: threshold fit by model
    :param minFitModelThreshold: When the fit is good, the minimum number of sample points required is treated as a threshold.
    :param debug:
    :param returnAll:
    :return:
    '''
    iterations = 0
    best_fit = None
    best_err = np.inf
    best_inlier_idxs = None
    all_points = len(data)
    if maxIteration == -1:
        maxIteration = 1000
    if minSample == -1:
        minSample = int(all_points * 0.5)
    if minFitModelThreshold == -1:
        minFitModelThreshold = int(all_points * 0.8)

    while iterations < maxIteration:
        maybe_idxs, test_idxs = random_partition(minSample, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]
        test_points = data[test_idxs]
        maybe_model = model.fit(maybe_inliers)
        test_err = model.getError(test_points, maybe_model)
        also_index = test_idxs[test_err < threshold]
        also_inliers = data[also_index, :]

        if debug:
            print("test_err.min: ", test_err.min())
            print("test_err.max: ", test_err.max())
            print("test_err.mean: ", np.mean(test_err))
            print ('iteration %d:len(alsoinliers) = %d' %(iterations, len(also_inliers)))

        if len(also_inliers) > minFitModelThreshold:
            better_data = np.concatenate((maybe_inliers, also_inliers))
            print(better_data.shape)
            better_model = model.fit(better_data)
            better_error = model.getError(better_data, better_model)
            this_err = np.mean(better_error)
            if this_err < best_err:
                best_fit = better_model
                best_err = this_err
                best_inlier_idxs = np.concatenate((maybe_idxs, also_index))

        iterations += 1
        if best_fit is None:
            raise ValueError("did't meet fit acceptance criteria")
        if returnAll:
            return best_fit,{'inliers':best_inlier_idxs}
        else:
            return best_fit

class SVDFitPlane():
    def __init__(self, inputCols, outputCols, debug=False):
        self.input_cols = inputCols
        self.output_cols = outputCols
        self.debug = debug

    def fit(self, data):
        xyz_mean = [np.mean(data[:, 0]), np.mean(data[:, 1]), np.mean(data[:, 2])]
        # print("xyz_mean: ", xyz_mean)
        point_vector = data - xyz_mean
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


    def getError(self, data, model):
        normal_vec = np.array(model[:3]).T
        ab = np.array(model[:2]).T
        c = model[2]
        d = model[3]
        D = data[:, 2]
        XY = data[:, :2]
        D_fit = (-1)*(np.dot(XY, ab) + d) / c
        # D_fit = np.dot(data, normal_vec)
        print("D: ", D)
        print("D_fit: ", D_fit)
        dist_per_point = ((D - D_fit) ** 2)
        return dist_per_point

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
    fit = SVDFitPlane(1, 1)
    plane_f = fit.fit(point_cloud_sample)
    err_list = fit.getError(point_cloud_sample, plane_f)
    print("err_list: ", err_list)
    # dist, dist_min, dist_max, dist_mean, dist_std = planeEvaluate(point_cloud_sample, plane_f)
    # print("dist_min: ", dist_min)
    # print("dist_max: ", dist_max)
    # print("dist_mean: ", dist_mean)
    # print("dist_std: ", dist_std)
    # calAverageError(plane_f, point_cloud_sample)
    # calAverageRMSE(plane_f, point_cloud_sample)

def ransacTest():
    point_cloud = np.loadtxt("/home/pi/Music/my_test_project/open3d/points_ahead.txt")
    model = SVDFitPlane(1, 1)
    ransac_fit = ransac(point_cloud, model, 4000, 10, 0.01, 2000)
    # ransac_fit = ransac(point_cloud, model)
    print("ransac_fit: ", ransac_fit)
# def main():
    # data = range(0, 21)
    # data = np.array(data).reshape(-1, 3)
    # print("data: ", data)
    # idx1, idx2 = random_partition(3, data.shape[0])
    #
    # print("idx1: ", idx1)
    # print("idx2: ", idx2)
    # points1 = data[idx1, :]
    # points2 = data[idx2]
    #
    # print("points1: ", points1)
    # print("points2: ", points2)



if __name__ == "__main__":
    # main()
    ransacTest()
    import cv2
