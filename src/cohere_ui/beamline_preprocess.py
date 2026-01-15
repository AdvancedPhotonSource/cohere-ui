#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This script reads raw data, applies correction related to instrument, aligns multiple scans data if applicable,
and saves the preprocessed data in cohere experiment space in the preprocessed_data subdirectory. User can control
some elements of the preprocess, like cropping area around maximum intensity, or accepting only scans with
set number of frames, removing outliers. The configuration parameters are described in :ref:`config_prep`.

It is designed to support different beamlines that may use different file format and different instrument.
The specifics are obtained from beamline instrument implementations. The configuration of instrument is define by
beamline. Refer to :ref:`config_instr` for definitions of these parameters.

To run this script from command line::

    beamline_preprocess <experiment_dir>

optional argument may follow:  --no_verify

One can use --help to get explanation of command line parameters.
"""

__author__ = "Barbara Frosik"
__docformat__ = 'restructuredtext en'
__all__ = ['handle_prep',
           'main']

import argparse
import importlib
import cohere_core.utilities as ut
import cohere_ui.api.preprocessor as preprocessor
import cohere_ui.api.common as com
import cohere_ui.api.multipeak as mp


def handle_prep(experiment_dir, **kwargs):
    """
    Reads the configuration files and accordingly to configuration creates prep_data.tif file in the
    <experiment_dir>/prep directory or for separate scans and for separate scan ranges creates multiple
    prep_data.tif files in <experiment_dir>/<scan_<scan_no>>/prep directories. It applies instrument
    corrections and aligns multiple scans data files.

    :param experimnent_dir: directory with experiment files
    :param kwargs:
        no_verify : boolean switch to determine if the verification error throws exception
    :return: experiment directory
    """
    print('pre-processing data')

    # requesting the configuration files that provide parameters for preprocessing
    conf_list = ['config_prep', 'config_instr', 'config_mp', 'config_data']
    conf_maps, converted = com.get_config_maps(experiment_dir, conf_list, **kwargs)

    # check the maps
    if 'config_instr' not in conf_maps.keys():
        print('exiting pre-processing')
        raise FileNotFoundError ('missing config_instr file, exiting')
    if 'config_prep' not in conf_maps.keys():
        print('info: no config_prep file, continuing')
        remove_outliers = False
    else:
        remove_outliers = conf_maps['config_prep'].get('remove_outliers', False)

    main_conf_map = conf_maps['config']

    # checked already if beamline is configured
    beamline = main_conf_map['beamline']
    try:
        instr_module = importlib.import_module(f'cohere_beamlines.{beamline}.instrument')
        need_detector = True # need to create for preprocessing
        instr_obj = instr_module.create_instr(conf_maps, need_detector=need_detector)
    except Exception as ex:
        print('exiting pre-processing')
        raise ex

    # get the settings from config
    separate_scans = main_conf_map.get('separate_scans', False)
    separate_scan_ranges = main_conf_map.get('separate_scan_ranges', False)
    multipeak = main_conf_map.get('multipeak', False)

    # get tuples of (scan, data info) for each scan in the scan ranges.
    # The scan_datainfo is a list of sub-lists, each sub-list reflects scan range.
    # If there is a single scan, the output will be: [[(scan, data info)]].
    #
    # Note: For aps_34idc the data info is a directory path to the data.
    # For esrf_id01 the data info is a node in hdf5 file.
    #
    # Note: Typically scans are represented by integer values. If a beamline defines scan in
    # another way, the parsing and creating scans_datainfo is encapsulated in the beamline instrument
    # object. The following logic still applies.
    scans_datainfo = instr_obj.datainfo4scans()

    if sum(len(inner_list) for inner_list in scans_datainfo) == 0:
        print('no data found for scans, exiting')
        return 'no data found for scans, exiting'

    outliers = []
    if separate_scans:
        # get all (scan, data info) tuples, process each scan and save the data in scans directories.
        single_scans_datainfo = [s_d for batch in scans_datainfo for s_d in batch]
        # passing in lambda: instr_obj.get_scan_array as the function is instrument dependent
        preprocessor.process_separate_scans(instr_obj.get_scan_array, single_scans_datainfo, experiment_dir)
    elif separate_scan_ranges:
        # combine scans within ranges, save the data in scan ranges directories.
        for batch in scans_datainfo:
            outliers.extend(preprocessor.process_batch(instr_obj.get_scan_array, batch, experiment_dir, separate_scan_ranges, remove_outliers))
    elif multipeak:
        outliers = mp.preprocess(instr_obj, scans_datainfo, experiment_dir, conf_maps)
    else:
        # combine all scans
        scans_datainfo = [e for batch in scans_datainfo for e in batch]
        outliers = preprocessor.process_batch(instr_obj.get_scan_array, scans_datainfo, experiment_dir, separate_scan_ranges, remove_outliers)

    # save configuration with the auto found outliers. Save even if no outliers found to show it.
    if 'config_prep' in conf_maps.keys():
        prep_conf_map = conf_maps['config_prep']
        if len(outliers) > 0:
            prep_conf_map['outliers_scans'] = outliers
        elif 'outliers_scans' in conf_maps['config_prep'].keys():
                # remove the outlies param
                conf_maps['config_prep'].pop('outliers_scans')
        ut.write_config(prep_conf_map, ut.join(experiment_dir, 'conf', 'config_prep'))

    print('finished beamline preprocessing')
    return ''


def main():
    """
    An entry function that takes command line parameters. It invokes the processing function handle_prep with
    the parameters. The command line parameters: experiment directory, --no_verify.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir",
                        help="directory where the configuration files are located")
    parser.add_argument("--no_verify", action="store_true",
                        help="if True the verifier has no effect on processing, error is always printed when incorrect configuration")
    parser.add_argument("--debug", action="store_true",
                        help="not used currently, available to developer for debugging")
    args = parser.parse_args()

    handle_prep(args.experiment_dir, no_verify=args.no_verify, debug=args.debug)


if __name__ == "__main__":
    exit(main())
