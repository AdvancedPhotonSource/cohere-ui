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
import cohere_core.data as fd
import cohere_core.utilities as ut
import inner_scripts.common as com


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['format_data',
           'main']


def format_data(experiment_dir, **kwargs):
    """
    For each prepared data in an experiment directory structure formats the data according to configured parameters and saves in the experiment space.

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
    print('formatting data')

    conf_list = ['config_data']
    err_msg, conf_maps, converted = com.get_config_maps(experiment_dir, conf_list, **kwargs)
    if len(err_msg) > 0:
        return err_msg

    # check the maps
    if 'config_data' not in conf_maps.keys():
        return 'missing config_data file'

    # verify that config files are correct
    main_conf_map = conf_maps['config']
    data_conf_map = conf_maps['config_data']

    no_center_max = data_conf_map.get('no_center_max', False)
    auto_data = main_conf_map.get('auto_data', False)
    data_conf_map['no_adjust_dims'] = True
    data_conf_map['no_center_max'] = True
    dfiles = []
    for scan_dir in os.listdir(experiment_dir):
        print('scan dir', scan_dir)
        if scan_dir.startswith('scan'):
            print('starts with scan', scan_dir)
            file_name = (ut.join(experiment_dir, scan_dir, 'preprocessed_data', 'prep_data.tif'))
            # the fd.prep function returns data_conf_map, as it can be updated if auto_data is True
            data_conf_map = fd.prep(file_name, auto_data, **data_conf_map)
            preprocessed_file_name = (ut.join(experiment_dir, scan_dir, 'phasing_data', 'data.tif'))
            dfiles.append(preprocessed_file_name)
            print('scan, size', scan_dir, os.path.getsize(ut.join(experiment_dir, scan_dir, 'preprocessed_data', 'prep_data.tif')))
    # assuming the first scan is full, followed by n low density scan, and so on.
    full_shape = ut.read_tif(dfiles[0]).shape
    small_shape = ut.read_tif(dfiles[1]).shape
    full_size = os.path.getsize(dfiles[0])

    # find fill_ratio, which means the pattern: full_size, (r - 1) small_size
    fill_ratio = int(full_shape[-1] / small_shape[-1] + .5)
    print('fill ratio', fill_ratio)

    # # find dimensions adjustment; it applies to all
    # pads = [((ut.get_good_dim(d) - d) // 2, ut.get_good_dim(d) - d - (ut.get_good_dim(d) - d) // 2) for d in full_shape]
    # c_vals = [(0.0, 0.0) for _ in range(len(full_shape))]

    for dfile in dfiles:
        data = ut.read_tif(dfile)
        if os.path.getsize(dfile) != full_size:
            full_data = np.full(full_shape, -1.0)
            no_frames = small_shape[-1]
            for i in range(no_frames):
                #print('filling slice', i * fill_ratio)
                full_data[:,:,i * fill_ratio] = data[:,:,i]
            data = full_data

        # the size still has to be adjusted to the optimal dimension
        crops_pads = kwargs.get('crop_pad', (0, 0, 0, 0, 0, 0))
        # adjust the size, either pad with 0s or crop array
        pairs = [crops_pads[2 * i:2 * i + 2] for i in range(int(len(crops_pads) / 2))]
        data = ut.adjust_dimensions(data, pairs)

        # do the centering now
        if not no_center_max:
            data, shift = ut.center_max(data)

        # remove temp tif file
        os.remove(dfile)
        # save the data
        np.save(dfile.split('.')[0], data)


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
