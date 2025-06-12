# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script create experiment directory space.

After the script is executed the experiment directory will contain "conf" subdirectory with configuration files. The initial configuration files contain all parameters, but most of them are commented out to clock the functionality.
"""

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['create_conf_prep',
           'create_conf_data',
           'create_conf_rec',
           'create_conf_disp',
           'create_exp',
           'main']

import argparse
import os
import cohere_core.utilities as ut
import cohere_ui.api.convertconfig as conv


def create_conf_prep(conf_dir):
    """
    Creates a "config_prep" file with some parameters commented out.

    Parameters
    ----------
    conf_dir : str
        directory where the file will be saved

    Returns
    -------
    nothing
    """
    conf_dir = conf_dir.replace(os.sep, '/')
    conf_file_name = conf_dir + '/config_prep'
    f = open(conf_file_name, "w+")

    f.write('data_dir = "/path/to/raw/data"\n')
    f.write('darkfield_filename = "/path/to/darkfield_file/dark.tif"\n')
    f.write('whitefield_filename = "/path/to/whitefield_file/dark.tif"\n')
    f.write('// roi = [0,256,0,256]\n')
    f.write('// min_files = 80\n')
    f.write('// exclude_scans = [78,81]\n')
    f.write('// Imult = 10000\n')
    f.close()


def create_conf_data(conf_dir):
    """
    Creates a "config_data" file with some parameters commented out.

    Parameters
    ----------
    conf_dir : str
        directory where the file will be saved

    Returns
    -------
    nothing
    """
    conf_dir = conf_dir.replace(os.sep, '/')
    conf_file_name = conf_dir + '/config_data'
    f = open(conf_file_name, "w+")

    f.write('// data_dir = "/path/to/dir/formatted_data/is/saved"\n')
    f.write('alien_alg = "none"\n')
    f.write('// aliens = [[170,220,112,195,245,123], [50,96,10,60,110,20]]\n')
    f.write('// aliens = "/path/to/maskfile/maskfile"\n')
    f.write('intensity_threshold = 20.0\n')
    f.write('// adjust_dimensions = [-13, -13, -65, -65, -65, -65]\n')
    f.write('// center_shift = [0,0,0]\n')
    f.write('// binning = [1,1,1]\n')
    f.close()


def create_conf_rec(conf_dir):
    """
    Creates a "config_rec" file with some parameters commented out.

    Parameters
    ----------
    conf_dir : str
        directory where the file will be saved

    Returns
    -------
    nothing
    """
    conf_dir = conf_dir.replace(os.sep, '/')
    conf_file_name = conf_dir + '/config_rec'
    f = open(conf_file_name, "w+")

    f.write('// data_dir = "/path/to/dir/with/formatted_data"\n')
    f.write('// save_dir = "/path/to/dir/to/save/results"\n')
    f.write('// init_guess = "random"\n')
    f.write('// processing = "auto"\n')
    f.write('reconstructions = 1\n')
    f.write('device = [0,1]\n')
    f.write('algorithm_sequence = "3* (20*ER + 180*HIO) + 20*ER"\n')
    f.write('hio_beta = .9\n')
    f.write('// ga_generations = 1\n')
    f.write('// ga_metrics = ["chi", "sharpness"]\n')
    f.write('// ga_breed_modes = ["sqrt_ab"]\n')
    f.write('// ga_cullings = [2,1]\n')
    f.write('// ga_sw_thresholds = [.15, .1]\n')
    f.write('// ga_sw_gauss_sigmas = [1.1, 1.0]\n')
    f.write('// ga_lpf_sigmas = [2.0, 1.5]\n')
    f.write('// ga_gen_pc_start = 3\n')
    f.write('twin_trigger = [2]\n')
    f.write('// twin_halves = [0, 0]\n')
    f.write('shrink_wrap_trigger = [10, 1]\n')
    f.write('shrink_wrap_type = "GAUSS"\n')
    f.write('shrink_wrap_threshold = 0.1\n')
    f.write('shrink_wrap_gauss_sigma = 1.0\n')
    f.write('initial_support_area = [.5,.5,.5]\n\n')
    f.write('// phm_trigger = [0, 1, 320]\n')
    f.write('// phm_phase_min = -1.57\n')
    f.write('// phm_phase_max = 1.57\n')
    f.write('// pc_interval = 50\n')
    f.write('// pc_type = "LUCY"\n')
    f.write('// pc_LUCY_iterations = 20\n')
    f.write('// pc_normalize = True\n')
    f.write('// pc_LUCY_kernel = [16,16,16]\n')
    f.write('// lowpass_filter_trigger = [0, 1, 320]\n')
    f.write('// lowpass_filter_sw_threshold = .1\n')
    f.write('// lowpass_filter_range = [.7]\n')
    f.write('// average_trigger = [-60, 1]\n')
    f.write('progress_trigger = [0, 20]')
    f.close()


def create_conf_disp(conf_dir):
    """
    Creates a "config_disp" file with some parameters commented out.

    Parameters
    ----------
    conf_dir : str
        directory where the file will be saved

    Returns
    -------
    nothing
    """
    conf_dir = conf_dir.replace(os.sep, '/')
    conf_file_name = conf_dir + '/config_disp'
    f = open(conf_file_name, "w+")

    f.write('// results_dir = "/path/to/dir/with/reconstructed/image(s)"\n')
    f.write('// rampups = 1\n')
    f.write('crop = [.5, .5, .5]\n')
    f.close()


def create_conf_disp(conf_dir):
    """
    Creates a "config_disp" file with some parameters commented out.

    Parameters
    ----------
    conf_dir : str
        directory where the file will be saved

    Returns
    -------
    nothing
    """
    conf_dir = conf_dir.replace(os.sep, '/')
    conf_file_name = conf_dir + '/config_instr'
    f = open(conf_file_name, "w+")

    f.write('diffractometer = "34idc"\n')
    f.write('// scanfile = "path/to/scanfile/scanfile"\n')
    f.close()


def create_exp(prefix, scan, working_dir, **args):
    """
    Concludes experiment name, creates directory, and "conf" subdirectory with initial configuration files.

    Parameters
    ----------
    prefix : str
        a literal ID of the experiment
    scan : str
        string indicating scan number, or scan range
        ex1: 5
        ex2: 67 - 89
    working_dir : str
        directory where the file will be saved
    specfile : str
        optional, name of specfile that was saved during the experiment

    Returns
    -------
    experiment_dir : str
        directory where the new experiment is located
    """
    id = prefix + '_' + scan
    working_dir = working_dir.replace(os.sep, '/')
    if not os.path.isdir(working_dir):
        print('working directory ' + working_dir + ' does not exist')
        return

    experiment_dir = working_dir + '/' + id
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    else:
        print('experiment with this id already exists')
        return experiment_dir

    experiment_conf_dir = experiment_dir + '/conf'
    if not os.path.exists(experiment_conf_dir):
        os.makedirs(experiment_conf_dir)

    # Based on params passed to this function create a temp config file and then copy it to the experiment dir.
    experiment_main_config = experiment_conf_dir + '/config'
    conf_map = {}
    conf_map['working_dir'] = working_dir
    conf_map['experiment_id'] = prefix
    conf_map['scan'] = scan
    if 'beamline' in args:
        conf_map['beamline'] = args['beamline']

    # get converter version
    conf_map['converter_ver'] = conv.get_version()
    conf_map['separate_scans'] = False
    conf_map['separate_scan_ranges'] = False

    ut.write_config(conf_map, experiment_main_config)

    # create simple configuration for each phase
    create_conf_prep(experiment_conf_dir)
    create_conf_data(experiment_conf_dir)
    create_conf_rec(experiment_conf_dir)
    create_conf_disp(experiment_conf_dir)

    return experiment_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("id", help="prefix to name of the experiment/data reconstruction")
    parser.add_argument("scan", help="a range of scans to prepare data from")
    parser.add_argument("working_dir", help="directory where the created experiment will be located")
    parser.add_argument('--beamline', action='store')
    parser.add_argument('--specfile', action='store')

    args = parser.parse_args()
    scan = args.scan
    id = args.id
    working_dir = args.working_dir

    varpar = {}
    if args.specfile and os.path.isfile(args.specfile):
        varpar['specfile'] = args.specfile

    if args.beamline:
        varpar['beamline'] = args.beamline

    return create_exp(id, scan, working_dir, **varpar)


if __name__ == "__main__":
    exit(main())