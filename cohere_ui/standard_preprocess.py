#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################
"""
This script reads all instrument corrected data instances found in the cohere experiment space.
The data is then preprocessed for phasing by calling the standard_preprocessing api from the cohere_core package.
Refer to :ref:`config_dt` for definition of parameters that direct the processing.

The steps implemented in standard_preprocessing: removing aliens, removing noise, applying sqrt, cropping and padding,
centering, and binning.

If running this script in **user mode** (i.e. after installing cohere_ui package with pypi), use this command:
    **standard_preprocess <experiment_dir>**

To run this script in **developer mode** (i.e. after cloning the cohere-ui repository) navigate to cohere-ui directory and
use the following command:
    **python cohere_ui/standard_preprocess.py <experiment_dir>**
optional argument may follow:  --no_verify

In either of the mode one can use --help to get explanation of command line parameters.
"""

import argparse
import os
import cohere_core.data as fd
import cohere_core.utilities as ut
import cohere_ui.api.common as com


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['format_data',
           'main']


def format_data(experiment_dir, **kwargs):
    """
    The script formats the data according to configured parameters and saves in the experiment space for each prepared
    data found in the in the experiment directory structure.

    :param experiment_dir: directory where the experiment files are saved
    :param kwargs: may contain:
        no_verify : boolean switch to determine if the verification error is returned
        debug : boolean switch not used in this code
    """
    print('formatting data')

    conf_list = ['config_data']
    conf_maps, converted = com.get_config_maps(experiment_dir, conf_list, **kwargs)
    if 'config_data' not in conf_maps: # not possible to get intensity threshold
        msg = 'Missing config_data file, cannot determine intensity threshold.'
        raise ValueError(msg)

    # check the maps
    data_conf_map = conf_maps.get('config_data')

    dirs = os.listdir(experiment_dir)
    for dir in dirs:
        if dir.startswith('scan') or dir.startswith('mp'):
            scan_dir = ut.join(experiment_dir, dir)
            data_dir = ut.join(scan_dir, 'phasing_data')
            proc_dir = scan_dir
        elif dir == 'preprocessed_data':
            data_dir = data_conf_map.get('data_dir', ut.join(experiment_dir, 'phasing_data'))
            proc_dir = experiment_dir
        else:
            continue

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # call the preprocessing in cohere_core, it will return updated configuration if auto_intensity_threshold is set
        data_conf_map = fd.prep(ut.join(proc_dir, 'preprocessed_data', 'prep_data.tif'), **data_conf_map)

    # This will work for a single reconstruction.
    # For separate scan the last auto-calculated values will be saved
    # TODO:
    # make the parameters like threshold a list for the separate scans scenario
    ut.write_config(data_conf_map, ut.join(experiment_dir,'conf', 'config_data'))


def main():
    """
    An entry function that takes command line parameters. It invokes the processing function format_data with
    the parameters. The command line parameters: experiment directory, --no_verify.
    """
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
