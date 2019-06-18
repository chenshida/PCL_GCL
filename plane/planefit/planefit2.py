import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def set_axes_radius(ax, origin, radius):
    # https://stackoverflow.com/a/50664367
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def solve(xyz, tol=1.0e-10):
    """ Solve for normal vector and regression error using la.lstsq
        Expects xyz to be a numpy array with the data 
        x, y, and z as column vectors
    """
    means = xyz.mean(0)
    Nml = xyz - means
    idx = np.argmin(Nml.max(0) - Nml.min(0))
    indices = [0, 1, 2]
    indices.remove(idx)
    M = Nml[:, indices]
    w = Nml[:, idx]

    result = opt.lsq_linear(M, w, tol=tol)
    x = result.x
    normal = np.insert(-x[:], idx, 1)
    return normal, means, result

def test(n=1000, noise=10, offset=50, tol=1.0e-10):
    """
        Test the least squares algorithm with a point cloud 
        with a given number of points, noise level and offset
    """
    p = (np.random.rand(3) - 0.5)
    p[0] = 0
    xyz = (np.random.rand(n, 3) - 0.5)*100
    xyz[:, 0] = xyz@p + (np.random.rand(n)-.5)*noise + offset
    normal, means, result = solve(xyz, tol=tol)
    print("normal vector 'normal': ", normal)
    print("center point 'means': ", means)
    print("least square solution 'p': ", result.x)
    print("least square solution 'cost': ", result.cost)
    print("LS residual squared 'fun': ", result.fun)
    print("Active_mask 'active_mask': ", result.active_mask)
    print("nit 'nit': ", result.nit)
    print("status 'status': ", result.status)
    print("msg 'message': ", result.message)
    print("success 'success': ", result.success)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    data = list(zip(means, means + normal * 10))
    ax.plot(*data, 'r')
    set_axes_equal(ax)
    plt.show()
