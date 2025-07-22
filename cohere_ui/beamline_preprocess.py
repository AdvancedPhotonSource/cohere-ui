#!/usr/bin/env python

# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script reads raw data, applies correction related to instrument, and saves prepared data.
This script is written for a specific APS beamline. It reads multiple raw data files in each scan directory, applies
darkfield and whitefield correction if applicable, creates 3D stack for each scan, then alignes and combines with
other scans.
"""

__author__ = "Barbara Frosik"
__docformat__ = 'restructuredtext en'
__all__ = ['handle_prep',
           'main']

import argparse
from multiprocessing import Process
import importlib
import cohere_core.utilities as ut
import cohere_ui.api.auto_data as ad
import cohere_ui.api.common as com
import cohere_ui.api.multipeak as mp


def handle_prep(experiment_dir, **kwargs):
    """
    Reads the configuration files and accrdingly creates prep_data.tif file in <experiment_dir>/prep directory or multiple
    prep_data.tif in <experiment_dir>/<scan_<scan_no>>/prep directories.
    Parameters
    ----------
    experimnent_dir : str
        directory with experiment files
    kwargs: ver parameters
        may contain:
        - rec_id : reconstruction id, pointing to alternate config
        - no_verify : boolean switch to determine if the verification error is returned
        - debug : boolean switch not used in this code
    Returns
    -------
    experimnent_dir : str
        directory with experiment files
    """
    print('pre-processing data')

    # requesting the configuration files that provide parameters for preprocessing
    conf_list = ['config_prep', 'config_instr', 'config_mp', 'config_data']
    conf_maps, converted = com.get_config_maps(experiment_dir, conf_list, **kwargs)

    # check the maps
    if 'config_instr' not in conf_maps.keys():
        return 'missing config_instr file, exiting'
    if 'config_prep' not in conf_maps.keys():
        print('info: no config_prep file, continuing')
        remove_scans = None
    else:
        remove_scans = conf_maps['config_prep'].get('remove_scans', None)

    main_conf_map = conf_maps['config']

    # checked already if beamline is configured
    beamline = main_conf_map['beamline']
    instr_module = importlib.import_module(f'cohere_ui.beamlines.{beamline}.instrument')
    preprocessor = importlib.import_module(f'cohere_ui.beamlines.{beamline}.preprocessor')

    # # combine parameters from the above configuration files into one dictionary
    # all_params = {k:v for d in conf_maps.values() for k,v in d.items()}

    need_detector = True # need to create for preprocessing
    instr_obj = instr_module.create_instr(conf_maps, need_detector=need_detector)

    # get the settings from config
    remove_outliers = conf_maps['config_prep'].get('remove_outliers', False)
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

    if len(scans_datainfo) == 0:
        print('no data found for scans, exiting')
        return

    # remove exclude_scans from the scans_dirs
    if remove_scans is not None:
        scans_datainfo = [[s_d for s_d in batch if s_d[0] not in remove_scans] for batch in scans_datainfo]

    if len(scans_datainfo) == 0:
        print('no data left after removing scans, exiting')
        return

    # remove_outliers should not be configured for separate scans
    if remove_outliers and not separate_scans:
        prep_conf_map = conf_maps.get('config_prep', {})
        # get all (scan, data info) tuples, process each scan and save the data in scans directories.
        single_scans_datainfo = [s_d for batch in scans_datainfo for s_d in batch]
        # passing in lambda: instr_obj.get_scan_array as the function is instrument dependent
        preprocessor.process_separate_scans(instr_obj.get_scan_array, single_scans_datainfo, experiment_dir)

        outliers_scans = ad.find_outlier_scans(experiment_dir, scans_datainfo, separate_scan_ranges or multipeak)
        if len(outliers_scans) > 0:
            # remove outliers_scans from the scans_dirs
            scans_datainfo = [[s_d for s_d in batch if s_d[0] not in outliers_scans] for batch in scans_datainfo]
        # save configuration with the auto found outliers. Save even if no outliers found to show it.
        prep_conf_map['outliers_scans'] = outliers_scans
        ut.write_config(prep_conf_map, ut.join(experiment_dir, 'conf', 'config_prep'))

    if separate_scans:
        # get all (scan, data info) tuples, process each scan and save the data in scans directories.
        single_scans_datainfo = [s_d for batch in scans_datainfo for s_d in batch]
        # passing in lambda: instr_obj.get_scan_array as the function is instrument dependent
        preprocessor.process_separate_scans(instr_obj.get_scan_array, single_scans_datainfo, experiment_dir)
    elif separate_scan_ranges:
        # combine scans within ranges, save the data in scan ranges directories.
        processes = []
        for batch in scans_datainfo:
            p = Process(target=preprocessor.process_batch,
                        args=(instr_obj.get_scan_array, batch, experiment_dir, separate_scan_ranges))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    elif multipeak:
        mp.preprocess(preprocessor, instr_obj, scans_datainfo, experiment_dir, conf_maps)
    else:
        # combine all scans
        scans_datainfo = [e for batch in scans_datainfo for e in batch]
        preprocessor.process_batch(instr_obj.get_scan_array, scans_datainfo, experiment_dir, separate_scan_ranges)

    print('finished beamline preprocessing')
    return ''


def main():
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
