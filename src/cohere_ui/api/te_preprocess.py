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


def format_data(experiment_dir, conf_maps):
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

    print('formatting data, chrono')

    data_conf_map = conf_maps['config_data']
    auto_data = data_conf_map.get('auto_intensity_threshold', False)
    intensity_threshold = data_conf_map.get('intensity_threshold', None)

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

    # The shape should be adjusted for the workable size
    scan_dim = ut.get_good_dim(full_no_frames, 'cp')

    # add slices filled with -1.0 in place of not collected frames in data files with partial data
    for dfile in dfiles:
        if dfile[0].shape[-1] != full_no_frames:
            full_data = np.full(full_shape, -1.0)
            for i in range(partial_no_frames):
                full_data[:,:,i * fill_ratio] = dfile[0][:,:,i]
            data = full_data
        else:
            data = dfile[0]
        # add slices at the end to the 'good dimension'
        if scan_dim != full_no_frames:
            pad_width = ((0, 0), (0, 0), (0, scan_dim - full_no_frames))
            data = np.pad(data, pad_width, mode='constant', constant_values=0)
        # adjust all dimensions to the 'good' values.
        data = ut.array_to_good_dims(data, 'cp')

        # the adjust dims, binning should not be done for chrono

        # save in npy format to keep -1
        scan_dir = dfile[1]
        np.save(ut.join(scan_dir, 'phasing_data', 'data.npy'), data)

    print('finished preprocessing')


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
