# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import os
import numpy as np
import pyvista as pv
import cohere_core.utilities as ut
import cohere_core.utilities.dvc_utils as dvut
import cohere_ui.api.auto_data as ad
import matplotlib.pyplot as plt
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


def set_lib_from_pkg(pkg):
    """
    Imports package specified in input and sets the device library to this package.

    :param pkg: supported values: 'cp' for cupy, 'np' for numpy, and 'torch' for torch
    :return:
    """
    global devlib

    # get the lib object
    devlib = ut.get_lib(pkg)
    # the utilities are not associated with reconstruction and the initialization of lib is independent
    dvut.set_lib_from_pkg(pkg)


def get_corr(arrays, cc_shift_dict, scans, pair):
    (i, j) = pair
    cc_shift_dict[scans[i]][scans[j]] = dvut.get_ccamax_cc(arrays[i], arrays[j])


def save_results4scan(scan, info, instrument, save_dir, do_RSM):
    set_lib_from_pkg('np')
    arr, offset = instrument.get_scan_array(info)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ut.save_tif(arr, ut.join(save_dir, 'prep_data.tif'))
    if do_RSM:
        rsmlab = instrument.get_RSM(scan)
        # save the rsm
        np.save(ut.join(save_dir, 'rsmlab.npy'), rsmlab)
        dl = pv.StructuredGrid(rsmlab[:, :, :, 0], rsmlab[:, :, :, 1], rsmlab[:, :, :, 2])
        dl.save(ut.join(save_dir, 'rsmlab.vts'))

    # process data info
    max_ind = [int(ind) for ind in devlib.unravel_index(devlib.argmax(arr), arr.shape)]
    # add offset obtain from get_scan_array (roi and/or max_crop)
    max_ind[0] = max_ind[0] + int(offset[0])
    max_ind[1] = max_ind[1] + int(offset[1])

    PeakQ = instrument.get_pixelQ(max_ind, int(scan))

    frame = pd.DataFrame([{'scan': scan,
                           'max ind (x)': max_ind[0],
                           'max ind (y)': max_ind[1],
                           'max ind frame': max_ind[2],
                           'max intensity': devlib.amax(arr),
                           'peak_qx': PeakQ[0],
                           'peak_qy': PeakQ[1],
                           'peak_qz': PeakQ[2],
                           'Q_mag': np.linalg.norm(PeakQ),
                           'd_spacing': 2 * np.pi / np.linalg.norm(PeakQ)}])
    with pd.ExcelWriter(ut.join(save_dir, 'preprocess.xlsx'), engine='xlsxwriter') as writer:
        frame.to_excel(writer, sheet_name='testSheetJ', startrow=0, startcol=0, index=False)
    print('preprocessed data shape ', arr.shape)


def process_batch(scans_infos, experiment_dir, separate_scan_ranges, remove_outliers, instrument, do_RSM):
    save_dir = ut.join(experiment_dir, 'preprocessed_data')
    # read the data in batch into memory
    if len(scans_infos) == 1:
        (scan, info) = scans_infos[0]
        if separate_scan_ranges:
            # save dir changes
            save_dir = ut.join(experiment_dir, f'scan_{str(scan)}', 'preprocessed_data')
        save_results4scan(scan, info, instrument, save_dir, do_RSM)
        return []

    # if more scans find correlation between them
    n = len(scans_infos)
    pkg = 'np'
    # try:
    #     import cupy as cp
    #     pkg = 'cp'
    #     # TODO
    #     # Hard coded for now, need to find formula
    #     no_proc = 10
    # except:
    #     no_proc = min(os.cpu_count(), n * n)
    set_lib_from_pkg(pkg)

    scans, info = zip(*scans_infos)
    scans = list(scans)
    arrays_offsets = [instrument.get_scan_array(scan_info) for scan_info in info]
    arrays = [array_offset[0] for array_offset in arrays_offsets]
    # for ar in arrays:
    #     ar =
    offsets = [array_offset[1] for array_offset in arrays_offsets]
    # check if shapes are the same. It is possible that max_crop if applied may return array size
    # smaller than the crop if the maximum is close to the edge of frame.
    shape = arrays[0].shape
    arrays = [ar for ar in arrays if ar.shape == shape]
    if len(arrays) != n:
        print(
            'Exiting preprocessing. Array shape mismatch. May be due to applying max_crop and maximum being close to the edge')
        return []

    # normalize
    normalized_arrays = [a / devlib.norm(a) for a in arrays]

    # get shifts to align and correlation(between each of the arrays)
    cc_shift_dict = {scan: {} for scan in scans}

    pairs = [(i, j) for i in list(range(n)) for j in list(range(n))]
    func = partial(get_corr, normalized_arrays, cc_shift_dict, scans)
    no_proc = min(os.cpu_count(), n * n)
    with ThreadPoolExecutor(max_workers=no_proc) as exe:
        exe.map(func, pairs)

    cc_matrix = np.array([[inner_dict[key][1] for key in scans] for inner_dict in cc_shift_dict.values()])

    # delete the normalized_arrays
    del normalized_arrays

    # show cross correlation including outliers
    fig, ax = plt.subplots()
    plt.title(f'Cross-correlation of scans')
    plt.imshow(cc_matrix)
    plt.colorbar()
    ax.set_xticks(range(n), labels=scans, rotation=90, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n), labels=scans)

    outliers = []
    if remove_outliers and len(scans) > 3:
        print('removing outliers')
        outliers = ad.find_outliers_in_batch(1.0 - cc_matrix, scans)
        to_remove = [i for i, scan in enumerate(scans) if scan in outliers]
        to_remove.reverse()
        for i in to_remove:
            scans.pop(i)
            offsets.pop(i)
            arrays.pop(i)

    if separate_scan_ranges:
        save_dir = ut.join(experiment_dir, f'scan_{str(scans[0])}-{str(scans[-1])}', 'preprocessed_data')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # The save_dir is determined, save the plot
    path = os.path.join(save_dir, 'cross_correlation.png')
    plt.savefig(path)

    # shift arrays to the first one and add the intensities
    sum_arr = arrays[0]
    shape = devlib.array(sum_arr.shape)
    first_scan = scans[0]
    shifts_cc = cc_shift_dict[first_scan]
    # cc_shift_dict items are (shifts, cc). Get the shifts only and convert to list of int for each scan.
    shifts = {k:np.array(v[0]).tolist() for k, v in shifts_cc.items()}
    correlations = {k:v[1] for k, v in shifts_cc.items()}

    for i, scan in enumerate(scans[1:]):
        arr = arrays[i]
        pixelshift = shifts[scan]
        shifted_arr = dvut.fast_shift(arr, pixelshift)
        sum_arr = sum_arr + shifted_arr

    # save the data file
    ut.save_tif(sum_arr, ut.join(save_dir, 'prep_data.tif'))

    if do_RSM:
        # calculate for the first scan
        rsmlab = instrument.get_RSM(scans[0])
        np.save(ut.join(save_dir, 'rsmlab.npy'), rsmlab)
        dl = pv.StructuredGrid(rsmlab[:, :, :, 0], rsmlab[:, :, :, 1], rsmlab[:, :, :, 2])
        dl.save(ut.join(save_dir, 'rsmlab.vts'))

    # Process data info
    # save outliers if any
    if len(outliers) > 0:
        info_dict = {'outliers': outliers}
        frame2 = pd.DataFrame([info_dict])
    else:
        frame2 = None

    # save info of arrays being added, such as max intensity and index of max intensity
    max_inds = [np.array(devlib.unravel_index(devlib.argmax(arr), arr.shape)).tolist() for arr in arrays]
    # add offset obtain from get_scan_array (roi and/or max_crop)
    for m, o in zip(max_inds, offsets):
        m[0] += int(o[0])
        m[1] += int(o[1])

    PeakQ = [instrument.get_pixelQ(max_inds[i], scans[i]) for i in range(len(scans))]

    frame1 = pd.DataFrame([{'scan': scans[i],
                            'max ind (x)': max_inds[i][0],
                            'max ind (y)': max_inds[i][1],
                            'max ind frame': max_inds[i][2],
                            'max intensity': devlib.amax(arrays[i]),
                            'peak_qx': PeakQ[i][0],
                            'peak_qy': PeakQ[i][1],
                            'peak_qz': PeakQ[i][2],
                            'Q_mag': np.linalg.norm(PeakQ[i]),
                            'd_spacing': 2 * np.pi / np.linalg.norm(PeakQ[i]),
                            'shift': shifts[scans[i]],
                            'correlation': correlations[scans[i]]
                            } for i in range(len(scans))])

    # save relation to the first scan, such as shift and cross correlation
    # frame3 = pd.DataFrame([{'scan': s, 'shift': shifts[s], 'correlation': correlations[s]} for s in scans])
    spacer = 3
    starting_at = 0
    with pd.ExcelWriter(ut.join(save_dir, 'preprocess.xlsx'), engine='xlsxwriter') as writer:
        frame1.to_excel(writer, sheet_name='DataInfo', startrow=starting_at, startcol=0, index=False)
        starting_at += len(frame1) + spacer
        if frame2 is not None:
            frame2.to_excel(writer, sheet_name='DataInfo', startrow=starting_at, startcol=0, index=False)

    print('preprocessed data shape ', sum_arr.shape)
    return outliers


def process_separate_scans(read_scan_func, scans_datainfo, save_dir, instr, do_RSM):
    for (scan, dinfo) in scans_datainfo:
        scan_save_dir = ut.join(save_dir, f'scan_{scan}', 'preprocessed_data')
        save_results4scan(scan, dinfo, instr, scan_save_dir, do_RSM)

