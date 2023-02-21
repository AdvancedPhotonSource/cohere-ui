# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script processes the multi-peak reconstructed image for visualization.
After the script is executed the experiment directory will contain image.vti file containing density, support, and the
three components of atomic displacement.
"""

__author__ = "Nick Porter"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['rotate_peaks',
           'twin_matrix']

import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.spatial.transform import Rotation as R

import util.util as ut
from beamlines.aps_34idc import beam_stuff as geo, diffractometers as diff, detectors as det


def rotate_peaks(prep_obj, data, scans, o_twin):
    print("rotating diffraction pattern")
    main_config = ut.read_config(prep_obj.experiment_dir + '/conf/config')
    main_config['last_scan'] = scans[-1]
    p = geo.Params(main_config)
    p.set_instruments(det.create_detector(p.detector), diff.create_diffractometer(p.diffractometer))
    shape = data.shape
    B_recip, _ = geo.set_geometry(shape, p, xtal=True)
    B_recip = np.stack([B_recip[1, :], B_recip[0, :], B_recip[2, :]])
    voxel_size = np.abs(np.linalg.det(B_recip))**(1/3)

    B_recip = o_twin @ B_recip

    x = np.arange(-shape[0] // 2, shape[0] // 2)  # define old grid
    y = np.arange(-shape[1] // 2, shape[1] // 2)
    z = np.arange(-shape[2] // 2, shape[2] // 2)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    old_grid = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])  # flatten grid into a 3xN array

    new_points = B_recip @ old_grid
    min_max = np.array([[np.min(row), np.max(row)] for row in new_points])

    a = np.arange(min_max[0, 0], min_max[0, 1], voxel_size)  # define new grid from the smallest to largest value
    b = np.arange(min_max[1, 0], min_max[1, 1], voxel_size)  # ensure that the step size is that of the unit cell
    c = np.arange(min_max[2, 0], min_max[2, 1], voxel_size)  # edge length
    aa, bb, cc = np.meshgrid(a, b, c, indexing='ij')

    new_grid = np.vstack([aa.ravel(), bb.ravel(), cc.ravel()])  # flatten new grid
    interp_points = np.linalg.inv(B_recip) @ new_grid  # transform new grid points to old grid

    interp = RGI((x, y, z), data, fill_value=0, bounds_error=False)  # feed interpolator node values from data
    data = interp(interp_points.T).reshape((a.shape[0], b.shape[0], c.shape[0]))  # interpolate values on new grid

    final_size = prep_obj.final_size
    shp = np.array([final_size, final_size, final_size]) // 2
    shp1 = np.array(data.shape) // 2
    pad = shp - shp1
    pad[pad < 0] = 0

    data = np.pad(data, [(pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2])])
    shp1 = np.array(data.shape) // 2
    start, end = shp1 - shp, shp1 + shp
    return data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]


def twin_matrix(hklin, hklout, twin_plane, sample_axis):
    r1 = ut.normalize(hklin)
    r3 = ut.normalize(np.cross(hklin, hklout))
    r2 = ut.normalize(np.cross(r3, r1))
    rmat = np.stack([r1, r2, r3])

    twin_plane = rmat @ twin_plane
    theta = np.arccos(twin_plane @ sample_axis / (np.linalg.norm(twin_plane)*np.linalg.norm(sample_axis)))
    vec = ut.normalize(np.cross(twin_plane, sample_axis))

    return R.from_rotvec(vec * -theta).as_matrix()
