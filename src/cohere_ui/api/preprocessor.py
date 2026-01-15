# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import os
import numpy as np
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
    (i,j) = pair
    cc_shift_dict[scans[i]][scans[j]] = dvut.get_ccamax_cc(arrays[i], arrays[j])


def process_batch(get_scan_func, scans_infos, experiment_dir, separate_scan_ranges, remove_outliers):
    save_dir = ut.join(experiment_dir, 'preprocessed_data')
    # read the data in batch into memory
    if len(scans_infos) == 1:
        arr = get_scan_func(scans_infos[0][1])
        if separate_scan_ranges:
            # save dir changes
            scan = str(scans_infos[0][0])
            save_dir = ut.join(experiment_dir, f'scan_{scan}','preprocessed_data')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        ut.save_tif(arr, ut.join(save_dir, 'prep_data.tif'))
        return []

    # if more scans find correlation
    n = len(scans_infos)
    pkg = 'np'
    try:
        import cupy as cp
        pkg = 'cp'
        # TODO
        # Hard coded for now, need to find formula
        no_proc = 10
    except:
        no_proc = min(os.cpu_count(), n * n)
    set_lib_from_pkg(pkg)

    scans, info = zip(*scans_infos)
    arrays = [get_scan_func(scan_info) for scan_info in info]

    # normalize
    normalized_arrays = [a / devlib.norm(a) for a in arrays]

    # get shifts to align and correlation(between each of the arrays)
    cc_shift_dict = {scan : {} for scan in scans}

    pairs = [(i,j) for i in list(range(n)) for j in list(range(n))]
    func = partial(get_corr, normalized_arrays, cc_shift_dict, scans)
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
    ax.set_xticks(range(n), labels=scans, rotation = 90, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n), labels=scans)

    outliers = []
    if remove_outliers and len(scans) > 3:
        print('removing outliers')
        outliers = ad.find_outliers_in_batch(1.0 - cc_matrix, scans)
        for i, scan in enumerate(scans):
            if scan in outliers:
                del arrays[i]
        scans = [scan for scan in scans if scan not in outliers]

    if separate_scan_ranges:
        save_dir = ut.join(experiment_dir, f'scan_{str(scans[0])}-{str(scans[-1])}','preprocessed_data')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # The save_dir is determined, save the plot
    path = os.path.join(save_dir,'cross_correlation.png')
    plt.savefig(path)

    # shift arrays to the first one and add the intensities
    sum_arr = arrays[0]
    shape = devlib.array(sum_arr.shape)
    first_scan = scans[0]
    shifts = cc_shift_dict[first_scan]

    for i, scan in enumerate(scans[1:]):
        arr = arrays[i]
        intshift = devlib.array(shifts[scan][0])
        pixelshift = devlib.where(intshift >= shape / 2, intshift - shape, intshift)
        shifted_arr = dvut.fast_shift(arr, pixelshift)
        sum_arr = sum_arr + shifted_arr

    # save outliers if any
    if len(outliers) > 0:
        info_dict = {'outliers' : outliers}
        frame1 = pd.DataFrame([info_dict])
    else:
        frame1 = None

    # save info of arrays being added, such as max intensity and index of max intensity
    frame2 = pd.DataFrame([{'scan' : scans[i],
                            'max intensity' : devlib.amax(arrays[i]),
                            'max intensity indx' : devlib.unravel_index(devlib.argmax(arrays[i]), shape)}
                           for i in range(len(scans))])

    # save relation to the first scan, such as shift and cross correlation
    frame3 = pd.DataFrame([{'scan' : s, 'shift' : shifts[s][0], 'correlation' : shifts[s][1]} for s in scans])
    spacer = 3
    starting_at = 1
    with pd.ExcelWriter(ut.join(save_dir, 'preprocess.xlsx'), engine='xlsxwriter') as writer:
        if frame1 is not None:
            frame1.to_excel(writer, sheet_name='testSheetJ', startrow=starting_at, startcol=0, index=False)
            starting_at += len(frame1) + spacer
        frame2.to_excel(writer, sheet_name='testSheetJ', startrow=starting_at, startcol=0, index=False)
        starting_at += len(frame2) + spacer
        frame3.to_excel(writer, sheet_name='testSheetJ', startrow=starting_at, startcol=0, index=False)

    # save the data file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    ut.save_tif(sum_arr, ut.join(save_dir, 'prep_data.tif'))

    return outliers


def process_separate_scans(read_scan_func, scans_datainfo, save_dir):
    for (scan, dinfo) in scans_datainfo:
        arr = read_scan_func(dinfo)
        scan_save_dir = ut.join(save_dir, f'scan_{scan}', 'preprocessed_data')
        if not os.path.exists(scan_save_dir):
            os.makedirs(scan_save_dir)
        ut.save_tif(arr, ut.join(scan_save_dir, 'prep_data.tif'))
