# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
viz_util
===============

This module is a suite of utility functions supporting visualization.
It supports 3D visualization.
"""
import os
import numpy as np
import scipy.ndimage as ndi
import math as m


__author__ = "Ross Harder"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['read_config',
           'write_config',
           'fast_shift',
           'shift_to_ref_array',
           'center',
           'crop_center',
           'get_zero_padded_centered',
           'sub_pixel_shift',
           'remove_ramp',
           'get_gpu_load',
           'get_gpu_distribution',
           'estimate_no_proc',
           ]


def read_config(config):
    """
    This function gets configuration file. It checks if the file exists and parses it into a dictionary.

    Parameters
    ----------
    config : str
        configuration file name

    Returns
    -------
    dict
        dictionary containing parsed configuration, None if the given file does not exist
    """
    import ast

    config = config.replace(os.sep, '/')
    if not os.path.isfile(config):
        print(config, 'is not a file')
        return None

    param_dict = {}
    input = open(config, 'r')
    line = input.readline()
    while line:
        # Ignore comment lines and move along
        line = line.strip()
        if line.startswith('//') or line.startswith('#'):
            line = input.readline()
            continue
        elif "=" in line:
            param, value = line.split('=')
            # do not replace in strings
            value = value.strip()
            if value.startswith('('):
                value = value.strip().replace('(','[').replace(')',']')
            param_dict[param.strip()] = ast.literal_eval(value)
        line = input.readline()
    input.close()
    return param_dict


def write_config(param_dict, config):
    """
    Writes configuration to a file.

    Parameters
    ----------
    param_dict : dict
        dictionary containing configuration parameters

    config : str
        configuration name theparameters will be written into
    """
    with open(config.replace(os.sep, '/'), 'w+') as f:
        f.truncate(0)
        for key, value in param_dict.items():
            if type(value) == str:
                value = '"' + value + '"'
            f.write(key + ' = ' + str(value) + os.linesep)


# supposedly this is faster than np.roll or scipy interpolation shift.
# https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def fast_shift(arr, shifty, fill_val=0):
    """
    Shifts array by given numbers for shift in each dimension.
    Parameters
    ----------
    arr : ndarray
        array to shift
    shifty : list
        a list of integer to shift the array in each dimension
    fill_val : float
        values to fill emptied space
    Returns
    -------
    ndarray
        shifted array
    """
    dims = arr.shape
    result = np.ones_like(arr)
    result *= fill_val
    result_slices = []
    arr_slices = []
    for n in range(len(dims)):
        if shifty[n] > 0:
            result_slices.append(slice(shifty[n], dims[n]))
            arr_slices.append(slice(0, -shifty[n]))
        elif shifty[n] < 0:
            result_slices.append(slice(0, shifty[n]))
            arr_slices.append(slice(-shifty[n], dims[n]))
        else:
            result_slices.append(slice(0, dims[n]))
            arr_slices.append(slice(0, dims[n]))
    result_slices = tuple(result_slices)
    arr_slices = tuple(arr_slices)
    result[result_slices] = arr[arr_slices]
    return result


def shift_to_ref_array(fft_ref, array):
    """
    Returns an array shifted to align with ref.

    Parameters
    ----------
    fft_ref : ndarray
        Fourier transform of reference array
    array : ndarray
        array to align with reference array
    Returns
    -------
    ndarray
        array shifted to be aligned with reference array

    """
    # get cross correlation and pixel shift
    fft_array = np.fft.fftn(array)
    cross_correlation = np.fft.ifftn(fft_ref * np.conj(fft_array))
    corelated = np.array(cross_correlation.shape)
    amp = np.abs(cross_correlation)
    intshift = np.unravel_index(amp.argmax(), corelated)
    shifted = np.array(intshift)
    pixelshift = np.where(shifted >= corelated / 2, shifted - corelated, shifted)
    shifted_arr = fast_shift(array, pixelshift)
    del cross_correlation
    del fft_array
    return shifted_arr


def center(image, support):
    """
    Shifts the image and support arrays so the center of mass is in the center of array.
    Parameters
    ----------
    image, support : ndarray, ndarray
        image and support arrays to evaluate and shift
    Returns
    -------
    image, support : ndarray, ndarray
        shifted arrays
    """
    shape = image.shape
    max_coordinates = list(np.unravel_index(np.argmax(image), shape))
    for i in range(len(max_coordinates)):
        image = np.roll(image, int(shape[i] / 2) - max_coordinates[i], i)
        support = np.roll(support, int(shape[i] / 2) - max_coordinates[i], i)

    com = ndi.center_of_mass(np.absolute(image) * support)
    # place center of mass in the center
    for i in range(len(shape)):
        image = np.roll(image, int(shape[i] / 2 - com[i]), axis=i)
        support = np.roll(support, int(shape[i] / 2 - com[i]), axis=i)

    # set center phase to zero, use as a reference
    phi0 = m.atan2(image.flatten().imag[int(image.flatten().shape[0] / 2)],
                   image.flatten().real[int(image.flatten().shape[0] / 2)])
    image = image * np.exp(-1j * phi0)

    return image, support


def crop_center(arr, new_shape):
    """
    This function crops the array to the new size, leaving the array in the center.
    The new_size must be smaller or equal to the original size in each dimension.

    Parameters
    ----------
    arr : ndarray
        the array to crop

    new_shape : tuple
        new size

    Returns
    -------
    cropped : ndarray
        the cropped array
    """
    shape = arr.shape
    cropped = arr
    for i in range(len(shape)):
        if new_shape[i] > shape[i]:
            print('error, cannot crop to a bigger size, returning original array')
            return arr
        crop_front = int((shape[i] - new_shape[i]) / 2)
        crop_end = crop_front + new_shape[i]
        splitted = np.split(cropped, [crop_front, crop_end], axis=i)
        cropped = splitted[1]

    return cropped


def get_zero_padded_centered(arr, new_shape):
    """
    This function pads the array with zeros to the new shape with the array in the center.

    Parameters
    ----------
    arr : ndarray
        the original array to be padded

    new_shape : tuple
        new dimensions

    Returns
    -------
    centered : ndarray
        the zero padded centered array
    """
    shape = arr.shape
    pad = []
    c_vals = []
    for i in range(len(new_shape)):
        pad.append((0, new_shape[i] - shape[i]))
        c_vals.append((0.0, 0.0))
    arr = np.lib.pad(arr, (pad), 'constant', constant_values=c_vals)

    centered = arr
    for i in range(len(new_shape)):
        centered = np.roll(centered, int((new_shape[i] - shape[i] + 1) / 2), i)

    return centered


def sub_pixel_shift(arr, shf):
    """
    Shifts pixels in a regularly sampled LR image with a subpixel precision according to local gradient.


    Parameters
    ----------
    arr : ndarray
        array to shift

    sft : list of floats
        shift in each dimension

    Returns
    -------
    ndarray
        shifted array
    """
    row_shift, col_shift, z_shift = shf
    buf2ft = np.fft.fftn(arr)
    shape = arr.shape
    Nr = np.fft.ifftshift(np.array(list(range(-int(np.floor(shape[0] / 2)), shape[0] - int(np.floor(shape[0] / 2))))))
    Nc = np.fft.ifftshift(np.array(list(range(-int(np.floor(shape[1] / 2)), shape[1] - int(np.floor(shape[1] / 2))))))
    Nz = np.fft.ifftshift(np.array(list(range(-int(np.floor(shape[2] / 2)), shape[2] - int(np.floor(shape[2] / 2))))))
    [Nc, Nr, Nz] = np.meshgrid(Nc, Nr, Nz)
    Greg = buf2ft * np.exp(
        1j * 2 * np.pi * (-row_shift * Nr / shape[0] - col_shift * Nc / shape[1] - z_shift * Nz / shape[2]))
    return np.fft.ifftn(Greg)

    # Nn = [np.fft.ifftshift(np.array(list(range(-int(np.floor(shape[i]/2)), shape[i] - int(np.floor(shape[i]/2))))))
    #                        for i in range(len(shape))]
    # mg = np.meshgrid(*Nn)
    # [Nc, Nr] = np.meshgrid(Nc, Nr) ??
    # mult = [shf[i] * mg[i] / shape[i] for i in range(len(shape))]
    # Greg = buf2ft * np.exp(-1j* 2 * np.pi * sum(mult))

    return np.fft.ifftn(Greg)

# a = np.array([[0.1,0.2,0.3],[0.4,0.4,0.4]])
# l = [.5, .5]
# sa = sub_pixel_shift(a, l)
# print(sa)

def remove_ramp(arr, ups=1):
    """
    Smooths image array (removes ramp) by applaying math formula.
    Parameters
    ----------
    arr : ndarray
        array to remove ramp
    ups : int
        upsample factor
    Returns
    -------
    ramp_removed : ndarray
        smoothed array
    """
    shape = arr.shape
    new_shape = [dim * ups for dim in shape]
    padded = get_zero_padded_centered(arr, new_shape)
    padded_f = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(padded)))
    com = ndi.center_of_mass(np.power(np.abs(padded_f), 2))
    sft = [new_shape[i] / 2.0 - com[i] + .5 for i in range(len(shape))]
    sub_pixel_shifted = sub_pixel_shift(padded_f, sft)
    ramp_removed_padded = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(sub_pixel_shifted)))
    ramp_removed = crop_center(ramp_removed_padded, arr.shape)

    return ramp_removed


def get_gpu_load(mem_size, ids):
    """
    This function is only used when running on Linux OS. The GPUtil module is not supported on Mac.
    This function finds available GPU memory in each GPU that id is included in ids list. It calculates
    how many reconstruction can fit in each GPU available memory.

    Parameters
    ----------
    mem_size : int
        array size

    ids : list
        list of GPU ids user configured for use

    Returns
    -------
    list
        list of available runs aligned with the GPU id list
    """
    import GPUtil

    gpus = GPUtil.getGPUs()
    total_avail = 0
    available_dir = {}
    for gpu in gpus:
        if gpu.id in ids:
            free_mem = gpu.memoryFree
            avail_runs = int(free_mem / mem_size)
            if avail_runs > 0:
                total_avail += avail_runs
                available_dir[gpu.id] = avail_runs
    available = []
    for id in ids:
        try:
            avail_runs = available_dir[id]
        except:
            avail_runs = 0
        available.append(avail_runs)
    return available


def get_gpu_distribution(runs, available):
    """
    Finds how to distribute the available runs to perform the given number of runs.

    Parameters
    ----------
    runs : int
        number of reconstruction requested

    available : list
        list of available runs aligned with the GPU id list

    Returns
    -------
    list
        list of runs aligned with the GPU id list, the runs are equally distributed across the GPUs
    """
    from functools import reduce

    all_avail = reduce((lambda x, y: x + y), available)
    distributed = [0] * len(available)
    sum_distr = 0
    while runs > sum_distr and all_avail > 0:
        # balance distribution
        for i in range(len(available)):
            if available[i] > 0:
                available[i] -= 1
                all_avail -= 1
                distributed[i] += 1
                sum_distr += 1
                if sum_distr == runs:
                    break
    return distributed


def estimate_no_proc(arr_size, factor):
    """
    Estimates number of processes the prep can be run on. Determined by number of available cpus and size
    of array.

    Parameters
    ----------
    arr_size : int
        size of array
    factor : int
        an estimate of how much memory is required to process comparing to array size

    Returns
    -------
    int
        number of processes
    """
    from multiprocessing import cpu_count
    import psutil

    ncpu = cpu_count()
    freemem = psutil.virtual_memory().available
    nmem = freemem / (factor * arr_size)
    # decide what limits, ncpu or nmem
    if nmem > ncpu:
        return ncpu
    else:
        return int(nmem)