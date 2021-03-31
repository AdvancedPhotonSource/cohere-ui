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
__all__ = ['import_beamline',
           'handle_prep'
           'main']

import argparse
import pylibconfig2 as cfg
import os
import sys
import importlib


def handle_prep(experiment_dir):
    """
    Reads the configuration files and accrdingly creates prep_data.tif file in <experiment_dir>/prep directory or multiple
    prep_data.tif in <experiment_dir>/<scan_<scan_no>>/prep directories.
    Parameters
    ----------
    experimnent_dir : str
        directory with experiment files
    Returns
    -------
    experimnent_dir : str
        directory with experiment files
    """
    # check cofiguration
    print ('preaparing data')
    try:
        main_conf_file = os.path.join(experiment_dir, *("conf", "config"))
        with open(main_conf_file, 'r') as f:
            main_conf_map = cfg.Config(f.read())
    except Exception as e:
        print('Please check the configuration file ' + main_conf_file + '. Cannot parse ' + str(e))
        return None
    try:
        beamline = main_conf_map.beamline
        try:
            prep = importlib.import_module('beamlines.' + beamline + '.prep')
            det = importlib.import_module('beamlines.' + beamline + '.detectors')
        except:
            print('cannot import beamlines.' + beamline + '.prep module.')
            return
    except AttributeError:
        print('Beamline must be configured in configuration file ' + main_conf_file)
        return None
    try:
        prep_conf_file = os.path.join(experiment_dir, *("conf", "config_prep"))
        with open(prep_conf_file, 'r') as f:
            prep_conf_map = cfg.Config(f.read())
    except Exception as e:
        print('Please check the configuration file ' + prep_conf_file + '. Cannot parse ' + str(e))
        return None
    try:
        data_dir = prep_conf_map.data_dir.strip()
        if not os.path.isdir(data_dir):
            print('data directory ' + data_dir + ' is not a valid directory')
            return None
    except:
        print('please provide data_dir in configuration file')
        return None

    # create BeamPrepData object defined for the configured beamline
    prep_obj = prep.BeamPrepData(experiment_dir, main_conf_map, prep_conf_map)

    # get directories from prep_obj
    dirs, indexes = prep_obj.get_dirs(data_dir=data_dir)
    if len(dirs) == 0:
        print('no data found')
        return None

    det_name = prep_obj.get_detector_name()
    if det_name is not None:
        det_obj = det.create_detector(det_name)
        if det_obj is not None:
            prep_obj.set_detector(det_obj, prep_conf_map)
        else:
            print('detector not created')
            return None
    prep_obj.prep_data(dirs, indexes)

    print('done with prep')
    return experiment_dir


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="directory where the configuration files are located")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    handle_prep(experiment_dir)


if __name__ == "__main__":
    exit(main(sys.argv[1:]))