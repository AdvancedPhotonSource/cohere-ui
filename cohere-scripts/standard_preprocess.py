# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################


"""
This script formats data for reconstruction according to configuration.
"""

import sys
import argparse
import os
import common as com
import cohere_core as cohere
import cohere_core.utilities as ut


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['format_data',
           'main']


def format_dir(exp_dir, data_dir, auto_data, data_conf_map):
    prep_file = com.join(exp_dir, 'preprocessed_data', 'prep_data.tif')
    data_conf_map['data_dir'] = data_dir
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return cohere.prep(prep_file, auto_data, **data_conf_map)


def format_data(experiment_dir, **kwargs):
    """
    For each prepared data in an experiment directory structure formats the data according to configured parameters and saves in the experiment space.

    Parameters
    ----------
    experiment_dir : str
        directory where the experiment processing files are saved

    Returns
    -------
    nothing
    """
    print('formatting data')

    debug = 'debug' in kwargs and kwargs['debug']
    conf_list = ['config_data']
    err_msg, conf_maps = com.get_config_maps(experiment_dir, conf_list, debug)
    if len(err_msg) > 0:
        return err_msg
    main_conf_map = conf_maps['config']

    auto_data = 'auto_data' in main_conf_map and main_conf_map['auto_data']

    data_conf_map = conf_maps['config_data']
    if debug:
        data_conf_map['debug'] = True

    if 'data_dir' in data_conf_map:
        data_dir = data_conf_map['data_dir']
    else:
        data_dir = com.join(experiment_dir, 'phasing_data')
    data_conf_map = format_dir(experiment_dir, data_dir, auto_data, data_conf_map)

    dirs = os.listdir(experiment_dir)
    for dir in dirs:
        if dir.startswith('scan') or dir.startswith('mp'):
            scan_dir = com.join(experiment_dir, dir)
            data_dir = com.join(scan_dir, 'phasing_data')
            data_conf_map = format_dir(scan_dir, data_dir, auto_data, data_conf_map)

    # This will work for a single reconstruction.
    # For separate scan the last auto-calculated values will be saved
    # TODO:
    # make the parameters like threshold a list for the separate scans scenario
    if auto_data:
        ut.write_config(data_conf_map, com.join(experiment_dir,'conf', 'config_data'))


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory")
    parser.add_argument("--debug", action="store_true",
                        help="if True the vrifier has no effect on processing")
    args = parser.parse_args()
    format_data(args.experiment_dir, debug=args.debug)


if __name__ == "__main__":
    main(sys.argv[1:])
