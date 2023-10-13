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
import convertconfig as conv
import cohere_core as cohere
import util.util as ut


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
    experiment_dir = experiment_dir.replace(os.sep, '/')
    # convert configuration files if needed
    main_conf = experiment_dir + "/conf/config"
    if os.path.isfile(main_conf):
        config_map = ut.read_config(main_conf)
        if config_map is None:
            print ("info: can't read " + main_conf + " configuration file")
            return None
    else:
        print("info: missing " + main_conf + " configuration file")
        return None
    auto_data = 'auto_data' in config_map and config_map['auto_data']

    if 'converter_ver' not in config_map or conv.get_version() is None or conv.get_version() < config_map['converter_ver']:
        conv.convert(experiment_dir + '/conf')
        # re-parse config
        config_map = ut.read_config(main_conf)

    er_msg = cohere.verify('config', config_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        debug = 'debug' in kwargs and kwargs['debug']
        if not debug:
            return None

    # read the config_data
    data_conf = experiment_dir + "/conf/config_data"
    if os.path.isfile(data_conf):
        config_map = ut.read_config(data_conf)
        if config_map is None:
            print ("info: can't read " + data_conf + " configuration file")
            return None
    else:
        print("info: missing " + data_conf + " configuration file")
        return None
    debug = 'debug' in kwargs and kwargs['debug']
    er_msg = cohere.verify('config_data', config_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        if not debug:
            return None

    if debug:
        config_map['debug'] = True

    print('formating data')
    prep_file = experiment_dir + '/preprocessed_data/prep_data.tif'
    if os.path.isfile(prep_file):
        if 'data_dir' in config_map:
            data_dir = config_map['data_dir']
        else:
            data_dir = experiment_dir + '/phasing_data'
            config_map['data_dir'] = data_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        conf_data_map = cohere.prep(prep_file, auto_data, **config_map)

    dirs = os.listdir(experiment_dir)
    for dir in dirs:
        if dir.startswith('scan') or dir.startswith('mp'):
            scan_dir = experiment_dir + '/' + dir
            prep_file = scan_dir + '/preprocessed_data/prep_data.tif'
            # ignore configured 'data_dir' as it can't be reused by multiple scans
            data_dir = scan_dir + '/phasing_data'
            config_map['data_dir'] = data_dir
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            conf_data_map = cohere.prep(prep_file, auto_data, **config_map)
            
    if auto_data:
        ut.write_config(conf_data_map, data_conf)


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory")
    parser.add_argument("--debug", action="store_true",
                        help="if True the vrifier has no effect on processing")
    args = parser.parse_args()
    format_data(args.experiment_dir, debug=args.debug)


if __name__ == "__main__":
    main(sys.argv[1:])
