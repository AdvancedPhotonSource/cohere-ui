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
import cohere_core.data as fd
import cohere_core.utilities as ut
import common as com


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

    Returns
    -------
    nothing
    """
    print('formatting data')

    conf_list = ['config_data']
    conf_maps, converted = com.get_config_maps(experiment_dir, conf_list)
     # check the maps
    if 'config' not in conf_maps.keys():
        return 'missing main config file'
    if 'config_data' not in conf_maps.keys():
        return 'missing config_data file'

    # verify that config files are correct
    main_conf_map = conf_maps['config']
    err_msg = ut.verify('config', main_conf_map)
    if len(err_msg) > 0:
        return err_msg

    data_config_map = conf_maps['config_data']
    err_msg = ut.verify('config_data', data_config_map)
    if len(err_msg) > 0:
        return err_msg

    auto_data = 'auto_data' in main_conf_map and main_conf_map['auto_data']

    data_conf_map = conf_maps['config_data']

    dirs = os.listdir(experiment_dir)
    for dir in dirs:
        if dir.startswith('scan') or dir.startswith('mp'):
            scan_dir = ut.join(experiment_dir, dir)
            data_dir = ut.join(scan_dir, 'phasing_data')
            proc_dir = scan_dir
        elif dir == 'preprocessed_data':
            if 'data_dir' in data_conf_map:
                data_dir = data_conf_map['data_dir']
            else:
                data_dir = ut.join(experiment_dir, 'phasing_data')
            proc_dir = experiment_dir
        else:
            continue

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # call the preprocessing in cohere_core, it will return updated configuration if auto_data
        data_conf_map = fd.prep(ut.join(proc_dir, 'preprocessed_data', 'prep_data.tif'), auto_data, **data_conf_map)

    # This will work for a single reconstruction.
    # For separate scan the last auto-calculated values will be saved
    # TODO:
    # make the parameters like threshold a list for the separate scans scenario
    if auto_data:
        ut.write_config(data_conf_map, ut.join(experiment_dir,'conf', 'config_data'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory")
    parser.add_argument("--no_verify", action="store_true",
                        help="if True the vrifier has no effect on processing")
    args = parser.parse_args()
    format_data(args.experiment_dir, no_verify=args.no_verify)


if __name__ == "__main__":
    main()
