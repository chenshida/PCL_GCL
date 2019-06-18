#!/usr/bin/python3
# -*- coding: utf-8 -*-
__Author__ = 'csd'
__version__ = "1.0"
__data__ = "2019.05.17"

import os
import sys
import numpy as np


class SVDFitPlane():
    def __init__(self):
        pass

    def fit(self, data):
        xyz_mean = [np.mean(data[:, 0]), np.mean(data[:, 1]), np.mean(data[:, 2])]
        point_vector = data - xyz_mean
        U, S, V = np.linalg.svd(point_vector)

        V = V.T
        a = V[0, 2]
        b = V[1, 2]
        c = V[2, 2]

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
    pass


if __name__ == "__main__":
    main()
