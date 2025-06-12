# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################


"""
This script formats data for reconstruction according to configuration.
"""

import argparse
import os
import numpy as np
import cohere_core.utilities as ut
import cohere_ui.api.common as com


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['format_data',
           'main']


def format_data(experiment_dir, **kwargs):
    """
    This script does standard preprocessing for series of data files collected during time evolving experiment.

    It does the intensity thresholding, and replacing data by it's square root (steps in standard preprocessing).
    Following is special operation for the time evolving case, i.e. filling the missing frames with -1.
    This will only affect the data files that collected the partial data. This is for the reconstruction
    process to distinguish between full and partial data.
    After the insertion the preprocessing continues with centering max and adjusting dimensions,
    and then binning.
    The data is saved in npy file, instead of tif file, so the negative values are preserved.

    Parameters
    ----------
    experiment_dir : str
        directory where the experiment processing files are saved
    kwargs: ver parameters
        may contain:
        - no_verify : boolean switch to determine if the verification error is returned
        - debug : boolean switch not used in this code

    Returns
    -------
    nothing
    """
    def pre_format(ar, auto_data, intensity_threshold):
        if auto_data:
            # the formula for auto threshold was found empirically, may be
            # modified in the future if more tests are done
            auto_threshold_value = 0.141 * ar[np.nonzero(ar)].mean().item() - 3.062
            intensity_threshold = max(2.0, auto_threshold_value)
            print(f'auto intensity threshold: {intensity_threshold}')
        # zero out the noise
        ar = np.where(ar <= intensity_threshold, 0.0, ar)
        # square root data
        return np.sqrt(ar)

    print('formatting data')

    conf_list = ['config_data']
    err_msg, conf_maps, converted = com.get_config_maps(experiment_dir, conf_list, **kwargs)
    if len(err_msg) > 0:
        return err_msg

    main_conf_map = conf_maps['config']
    auto_data = main_conf_map.get('auto_data', False)

    # check the config data
    if 'config_data' not in conf_maps.keys():
        # it can still process if auto_data is set
        if auto_data:
            intensity_threshold = None
            no_center_max = False
        else:
            return 'missing config_data file'
    else:
        data_conf_map = conf_maps['config_data']
        intensity_threshold = data_conf_map.get('intensity_threshold', None)
        no_center_max = data_conf_map.get('no_center_max', None)

    # Find scan directories, read the data, and apply pre-format, i.e. threshold and sqroot
    # Store the data in a list, each scan data as tuple (data, scan dir, scan number)
    dfiles = []
    for dir in os.listdir(experiment_dir):
        if dir.startswith('scan'):
            scan_dir = ut.join(experiment_dir, dir)
            prep_data = ut.read_tif(ut.join(scan_dir, 'preprocessed_data', 'prep_data.tif'))
            data = pre_format(prep_data, auto_data, intensity_threshold)
            # add the tuple of (data, scan dir, scan number) to dfiles list
            dfiles.append((data, scan_dir, int(dir.split('_')[-1])))

            # create directory where the preprocessed data will be saved
            data_dir = ut.join(scan_dir, 'phasing_data')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

    # order data files by scan number
    dfiles = sorted(dfiles, key=lambda x: x[2])

    # save a list of scan directories with phasing data
    phasing_dirs = ut.join(experiment_dir, 'phasing_dirs')
    with open(phasing_dirs, 'w+') as pd:
        pd.write(str([dfile[1] for dfile in dfiles]))

    # The last dimension will be different in full data and partial data
    # find the last dimensions of two different arrays and calculate ratio of frames.
    # assuming the first scan is full.
    full_shape = dfiles[0][0].shape
    full_no_frames = dfiles[0][0].shape[-1]
    idx = 1
    while dfiles[idx][0].shape[-1] == full_no_frames:
        idx += 1
    partial_no_frames = dfiles[idx][0].shape[-1]

    # find fill_ratio
    fill_ratio = int(full_no_frames / partial_no_frames + .5)

    # add slices filled with -1.0 in place of not collected frames in data files with partial data
    for dfile in dfiles:
        if dfile[0].shape[-1] != full_no_frames:
            full_data = np.full(full_shape, -1.0)
            for i in range(partial_no_frames):
                full_data[:,:,i * fill_ratio] = data[:,:,i]
            data = full_data
        else:
            data = dfile[0]

        print('dfile full', dfile[1],(data < 0).sum() == 0)

        # even with crops_pads not given the size still has to be adjusted to the optimal dimension
        crops_pads = kwargs.get('crop_pad', (0, 0, 0, 0, 0, 0))
        # adjust the size, either pad with 0s or crop array
        pairs = [crops_pads[2 * i:2 * i + 2] for i in range(int(len(crops_pads) / 2))]
        data = ut.adjust_dimensions(data, pairs)

        # do the centering now
        if not no_center_max:
            data, shift = ut.center_max(data)

        # save in npy format to keep -1
        scan_dir = dfile[1]
        np.save(ut.join(scan_dir, 'phasing_data', 'data.npy'), data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory")
    parser.add_argument("--no_verify", action="store_true",
                        help="if True the verifier has no effect on processing, error is always printed when incorrect configuration")
    parser.add_argument("--debug", action="store_true",
                        help="not used currently, available to developer for debugging")
    args = parser.parse_args()
    format_data(args.experiment_dir, no_verify=args.no_verify, debug=args.debug)


if __name__ == "__main__":
    main()
