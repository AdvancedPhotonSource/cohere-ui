#!/usr/local/bin/python2.7.3 -tttt
"""
Functions to cut and plot the result RSM.
Created on Thu Apr 27 13:50:07 2023

@author: renzhe
@email: renzhe@ihep.ac.cn
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements
import os


def check_cut_box_size(bs, peak_pos, data_shape):
    """
    Check the box size possible for the symmetrical cut around the given peak position in the data.

    Parameters
    ----------
    bs : list
        The Half width of the boxsize in [Z, Y, X] order.
    peak_pos : list
        The peak position in the dataset.
    data_shape : Unions(list|tuple)
        The shape of the data to be cutted.

    Returns
    -------
    bs : list
        The suggested box size according to the shape of the data.

    """
    bs[0] = int(np.amin([bs[0], peak_pos[0] * 0.95, 0.95 * (data_shape[0] - peak_pos[0])]))
    bs[1] = int(np.amin([bs[1], peak_pos[1] * 0.95, 0.95 * (data_shape[1] - peak_pos[1])]))
    bs[2] = int(np.amin([bs[2], peak_pos[2] * 0.95, 0.95 * (data_shape[2] - peak_pos[2])]))
    return bs


def Cut_central(dataset, bs, cut_mode='maximum integration', peak_pos=None):
    """
    Cut the three dimensional dataset symmetrically with the box size given.

    Parameters
    ----------
    dataset : ndarray
        The three dimensional dataset to be cutted.
    bs : list
        The half width of the box size.
    cut_mode : str, optional
        The mode for choosing the center position for the cutting.
        The cut mode can be 'maximum integration', 'maximum intensity', 'weight center' and 'given'.
        'maximum integration': The dataset is cutted around the maximum integrated intensity in each dimension.
        'maximum intensity': The dataset is cutted around the maximum intensity.
        'weight center': The dataset is cutted around weight center in the box.
        'given': The dataset is cutted around the given peak position.
        The default is 'maximum integration'.
    peak_pos : list, optional
        The position for given peak position. The default is None.

    Returns
    -------
    intcut : ndarray
        The symmetrically cutted intensity.
    peak_pos : list
        The position of the peak position.
    bs : list
        The used box size for symmetrically cutting the dataset.

    """
    # Cutting the three dimensional data with the center of mass in the center of the intensity distribution
    if cut_mode == 'maximum integration':
        peak_pos = np.array([np.argmax(np.sum(dataset, axis=(1, 2))), np.argmax(np.sum(dataset, axis=(0, 2))), np.argmax(np.sum(dataset, axis=(0, 1)))], dtype=int)
        print('finding the centeral position for the cutting')
        bs = check_cut_box_size(bs, peak_pos, dataset.shape)
        intcut = np.array(dataset[(peak_pos[0] - bs[0]):(peak_pos[0] + bs[0]), (peak_pos[1] - bs[1]):(peak_pos[1] + bs[1]), (peak_pos[2] - bs[2]):(peak_pos[2] + bs[2])])
    elif cut_mode == 'maximum intensity':
        peak_pos = np.unravel_index(np.argmax(dataset), dataset.shape)
        bs = check_cut_box_size(bs, peak_pos, dataset.shape)
        intcut = np.array(dataset[(peak_pos[0] - bs[0]):(peak_pos[0] + bs[0]), (peak_pos[1] - bs[1]):(peak_pos[1] + bs[1]), (peak_pos[2] - bs[2]):(peak_pos[2] + bs[2])])
    elif cut_mode == 'weight center':
        peak_pos = np.array(np.around(measurements.center_of_mass(dataset)), dtype=int)
        bs = check_cut_box_size(bs, peak_pos, dataset.shape)
        intcut = np.array(dataset[(peak_pos[0] - bs[0]):(peak_pos[0] + bs[0]), (peak_pos[1] - bs[1]):(peak_pos[1] + bs[1]), (peak_pos[2] - bs[2]):(peak_pos[2] + bs[2])])
        print('cut according to the weight center')
        i = 0
        torlerence = 0.5
        while not np.allclose(measurements.center_of_mass(intcut), np.array(bs, dtype=float) - 0.5, atol=torlerence):
            peak_pos = np.array(peak_pos + np.around(measurements.center_of_mass(intcut) - np.array(bs, dtype=float) + 0.5), dtype=int)
            bs = check_cut_box_size(bs, peak_pos, dataset.shape)
            intcut = np.array(dataset[(peak_pos[0] - bs[0]):(peak_pos[0] + bs[0]), (peak_pos[1] - bs[1]):(peak_pos[1] + bs[1]), (peak_pos[2] - bs[2]):(peak_pos[2] + bs[2])])
            i += 1
            if i == 5:
                print("Loosen the constrain for the weight center cutting")
                torlerence = 1
            elif i > 8:
                print("could not find the weight center for the cutting")
                break
    elif cut_mode == 'given':
        if peak_pos is None:
            print('Could not find the given position for the cutting, please check it again!')
            peak_pos = np.array((dataset.shape) / 2, dtype=int)
        else:
            peak_pos = np.array(peak_pos, dtype=int)
        bs = check_cut_box_size(bs, peak_pos, dataset.shape)
        intcut = np.array(dataset[(peak_pos[0] - bs[0]):(peak_pos[0] + bs[0]), (peak_pos[1] - bs[1]):(peak_pos[1] + bs[1]), (peak_pos[2] - bs[2]):(peak_pos[2] + bs[2])])
    return intcut, peak_pos, bs


def plot_with_units(RSM_int, q_origin, unit, pathsavetmp, qmax=np.array([]), display_range=None):
    """
    Plot and save the diffraction pattern with correct units.

    Parameters
    ----------
    RSM_int : ndarray
        The diffraction pattern to be plotted.
    q_origin : list
        The minimum origin of the diffraction pattern.
    unit : float
        The unit of the diffraction pattern.
    pathsavetmp : str
        The template for saving the diffraction pattern.
        The parameter should be the complete path with %s in the filename for the position indicating different cut directions.
    qmax : list, optional
        If given, the cutted diffraction pattern will be displayed around the given position.
        Else the integrated diffraction intensity will be plotted.
        The default is np.array([]).
    display_range : list, optional
        The half width of the display range in [qz, qy, qx] order. The default is None.

    Returns
    -------
    None.

    """
    dz, dy, dx = RSM_int.shape
    qz = np.arange(dz) * unit + q_origin[0]
    qy = np.arange(dy) * unit + q_origin[1]
    qx = np.arange(dx) * unit + q_origin[2]
    # save the qx qy qz cut of the 3D intensity
    print('Saving the qx qy qz cuts......')
    plt.figure(figsize=(12, 12))
    pathsaveimg = pathsavetmp % ('qz')
    if len(qmax) == 0:
        plt.contourf(qx, qy, np.log10(np.sum(RSM_int, axis=0) + 1.0), 150, cmap='jet')
    else:
        plt.contourf(qx, qy, np.log10(RSM_int[qmax[0], :, :] + 1.0), 150, cmap='jet')
    plt.xlabel(r'Q$_x$ ($1/\AA$)', fontsize=20, fontstyle='italic', fontfamily='Arial', fontweight='bold')
    plt.ylabel(r'Q$_y$ ($1/\AA$)', fontsize=20, fontstyle='italic', fontfamily='Arial', fontweight='bold')
    plt.axis('scaled')
    plt.tick_params(axis='both', labelsize=20)
    if (display_range is not None) and (len(qmax) != 0):
        plt.xlim(qmax[2] * unit + q_origin[2] - display_range[2], qmax[2] * unit + q_origin[2] + display_range[2])
        plt.ylim(qmax[1] * unit + q_origin[1] - display_range[1], qmax[1] * unit + q_origin[1] + display_range[1])
    plt.savefig(pathsaveimg)
    plt.show()
    # plt.close()

    plt.figure(figsize=(12, 12))
    pathsaveimg = pathsavetmp % ('qy')
    if len(qmax) == 0:
        plt.contourf(qx, qz, np.log10(np.sum(RSM_int, axis=1) + 1.0), 150, cmap='jet')
    else:
        plt.contourf(qx, qz, np.log10(RSM_int[:, qmax[1], :] + 1.0), 150, cmap='jet')
    plt.xlabel(r'Q$_x$ ($1/\AA$)', fontsize=20, fontstyle='italic', fontfamily='Arial', fontweight='bold')
    plt.ylabel(r'Q$_z$ ($1/\AA$)', fontsize=20, fontstyle='italic', fontfamily='Arial', fontweight='bold')
    plt.axis('scaled')
    plt.tick_params(axis='both', labelsize=20)
    if (display_range is not None) and (len(qmax) != 0):
        plt.xlim(qmax[2] * unit + q_origin[2] - display_range[2], qmax[2] * unit + q_origin[2] + display_range[2])
        plt.ylim(qmax[0] * unit + q_origin[0] - display_range[0], qmax[0] * unit + q_origin[0] + display_range[0])
    plt.savefig(pathsaveimg)
    plt.show()
    # plt.close()

    plt.figure(figsize=(12, 12))
    pathsaveimg = pathsavetmp % ('qx')
    if len(qmax) == 0:
        plt.contourf(qy, qz, np.log10(np.sum(RSM_int, axis=2) + 1.0), 150, cmap='jet')
    else:
        plt.contourf(qy, qz, np.log10(RSM_int[:, :, qmax[2]] + 1.0), 150, cmap='jet')
    plt.xlabel(r'Q$_y$ ($1/\AA$)', fontsize=20, fontstyle='italic', fontfamily='Arial', fontweight='bold')
    plt.ylabel(r'Q$_z$ ($1/\AA$)', fontsize=20, fontstyle='italic', fontfamily='Arial', fontweight='bold')
    plt.axis('scaled')
    plt.tick_params(axis='both', labelsize=20)
    if (display_range is not None) and (len(qmax) != 0):
        plt.xlim(qmax[1] * unit + q_origin[1] - display_range[1], qmax[1] * unit + q_origin[1] + display_range[1])
        plt.ylim(qmax[0] * unit + q_origin[0] - display_range[0], qmax[0] * unit + q_origin[0] + display_range[0])
    plt.savefig(pathsaveimg)
    plt.show()
    # plt.close()
    return


def plot_without_units(RSM_int, mask, pathsavetmp):
    """
    Plot and save the diffraction pattern without units.

    Parameters
    ----------
    RSM_int : ndarray
        The diffraction pattern to be plotted.
    mask : ndarray
        The mask to be used. The masked pixels will be displayed by the red colors in the result plot.
    pathsavetmp : str
        The template for saving the diffraction pattern.
        The parameter should be the complete path with %s in the filename for the position indicating different cut directions.

    Returns
    -------
    None.

    """
    mask = np.ma.masked_where(mask == 0, mask)
    dz, dy, dx = RSM_int.shape
    # save the qx qy qz cut of the 3D intensity
    print('Saving the qx qy qz cuts......')
    plt.figure(figsize=(12, 12))
    pathsaveimg = pathsavetmp % 'qz'
    plt.imshow(np.log10(RSM_int[int(dz / 2), :, :] + 1.0), cmap='Blues')
    if mask.ndim != 1:
        plt.imshow(mask[int(dz / 2), :, :], cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    plt.xlabel(r'Q$_x$ (pixel)', fontsize=24)
    plt.ylabel(r'Q$_y$ (pixel)', fontsize=24)
    plt.axis('scaled')
    plt.tick_params(axis='both', labelsize=24)
    plt.savefig(pathsaveimg)
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 12))
    pathsaveimg = pathsavetmp % 'qy'
    plt.imshow(np.log10(RSM_int[:, int(dy / 2), :] + 1.0), cmap='Blues')
    if mask.ndim != 1:
        plt.imshow(mask[:, int(dy / 2), :], cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    plt.xlabel(r'Q$_x$ (pixel)', fontsize=24)
    plt.ylabel(r'Q$_z$ (pixel)', fontsize=24)
    plt.axis('scaled')
    plt.tick_params(axis='both', labelsize=24)
    plt.savefig(pathsaveimg)
    plt.show()
    plt.close()

    plt.figure(figsize=(12, 12))
    pathsaveimg = pathsavetmp % 'qx'
    plt.imshow(np.log10(RSM_int[:, :, int(dx / 2)] + 1.0), cmap='Blues')
    if mask.ndim != 1:
        plt.imshow(mask[:, :, int(dx / 2)], cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    plt.xlabel(r'Q$_y$ (pixel)', fontsize=24)
    plt.ylabel(r'Q$_z$ (pixel)', fontsize=24)
    plt.axis('scaled')
    plt.tick_params(axis='both', labelsize=24)
    plt.savefig(pathsaveimg)
    plt.show()
    plt.close()
    return


def RSM2vti(pathsave, RSM_dataset, filename, RSM_unit, origin=(0, 0, 0)):
    """
    Save the reciprocal space map to vti format for reading with paraview.

    Parameters
    ----------
    pathsave : str
        The folder path to save the RSM.
    RSM_dataset : ndarray
        The RSM to be saved.
    filename : str
        The filename for the saving.
    RSM_unit : float
        The unit of the RSM.
    origin : list, optional
        The origin of the RSM. The default is (0, 0, 0).

    Returns
    -------
    None.

    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    pathsave = os.path.join(pathsave, filename)
    imdata = vtk.vtkImageData()
    imdata.SetOrigin(origin[0], origin[1], origin[2])
    imdata.SetSpacing(RSM_unit, RSM_unit, RSM_unit)
    imdata.SetDimensions(RSM_dataset.shape)

    RSM_vtk = numpy_to_vtk(np.ravel(np.transpose(np.log10(RSM_dataset + 1.0))), deep=True, array_type=vtk.VTK_DOUBLE)

    imdata.GetPointData().SetScalars(RSM_vtk)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(pathsave)
    writer.SetInputData(imdata)

    writer.Write()
    return
