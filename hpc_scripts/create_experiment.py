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
__all__ = ['create_exp',
           'main']

import argparse
import os
import cohere_core.utilities as ut


def write_config(param_dict, config):
    """
    Writes configuration to a file.
    Parameters
    ----------
    param_dict : dict
        dictionary containing configuration parameters
    config : str
        configuration name theparameters will be written into
    """
    with open(config.replace(os.sep, '/'), 'w+') as f:
        f.truncate(0)
        linesep = os.linesep
        for key, value in param_dict.items():
            if type(value) == str:
                value = f'"{value}"'
            f.write(f'{key} = {str(value)}{linesep}')


def create_conf_main(conf_dir, working_dir, id, scan, beamline):
    conf_map = {}
    conf_map['working_dir'] = working_dir
    conf_map['experiment_id'] = id
    conf_map['scan'] = scan
    conf_map['beamline'] = beamline
    conf_map['converter_ver'] = 1
    conf_map['auto_data'] = True
    # TODO in the future it will support separate scans, separate scan ranges, multipeak experiments as well
    write_config(conf_map, conf_dir + '/config')


def create_conf_instr(conf_dir, specfile, diffractometer):
    conf_map = {}
    conf_map['specfile'] = specfile
    conf_map['diffractometer'] = diffractometer
    write_config(conf_map, ut.join(conf_dir, 'config_instr'))


def create_conf_prep(conf_dir, data_dir, darkfield_filename, whitefield_filename):
    conf_map = {}
    conf_map['data_dir'] = data_dir
    conf_map['darkfield_filename'] = darkfield_filename
    conf_map['whitefield_filename'] = whitefield_filename
    write_config(conf_map, ut.join(conf_dir, 'config_prep'))


def create_conf_data(conf_dir):
    conf_map = {}
    conf_map['intensity_threshold'] = 2.0
    write_config(conf_map, ut.join(conf_dir, 'config_data'))


def create_conf_rec(conf_dir):
    conf_map = {}
    conf_map['reconstructions'] = 10
    conf_map['device'] = [0, 1, 2, 3, 4, 5, 6]
    conf_map['algorithm_sequence'] = "1*(20*ER+80*HIO)+20*ER"
    conf_map['initial_support_area'] = [0.5, 0.5, 0.5]
    conf_map['shrink_wrap_trigger'] = [1, 1]
    conf_map['twin_trigger'] = [2]
    conf_map['ga_generations'] = 5
    conf_map['ga_fast'] = True
    write_config(conf_map, ut.join(conf_dir, 'config_rec'))


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
    conf_map = {}
    conf_map['crop'] = [.5, .5, .5]
    write_config(conf_map, ut.join(conf_dir, 'config_disp'))


def create_exp(working_dir, id, scan, beamline, data_dir, darkfield_filename, whitefield_filename,
               specfile, diffractometer):
    """
    Concludes experiment name, creates directory, and "conf" subdirectory with initial configuration files.

    Parameters
    ----------
    working_dir : str
        directory where the file will be saved
    id : str
        arbitrary name of cohere experiment
    scan : str
        string indicating scan number, or scan range(s)
    beamline : str
        name of the beamline recognized by cohere (for example: aps_34idc)
    darkfield_filename : str
        name of dark field file that applies to this experiment
    whitefield_filename : str
        name of white field file that applies to this experiment
    data_dir : str
        directory where raw data is collected
    specfile : str
        name of specfile that was saved during the experiment
    diffractometer : str
        name of diffractometer used in the experiment recognized by cohere (for example: 34idc)

    Returns
    -------
    nothing
    """
    experiment_dir = ut.join(working_dir, f'{id}_{scan}')
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    else:
        print('experiment with this id already exists')
        return

    experiment_conf_dir = experiment_dir + '/conf'
    os.makedirs(experiment_conf_dir)

    # create simple configuration for each phase
    create_conf_main(experiment_conf_dir, working_dir, id, scan, beamline)
    create_conf_instr(experiment_conf_dir, specfile, diffractometer)
    create_conf_prep(experiment_conf_dir, data_dir, darkfield_filename, whitefield_filename)
    create_conf_data(experiment_conf_dir)
    create_conf_rec(experiment_conf_dir)
    create_conf_disp(experiment_conf_dir)

    return experiment_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", required=True, help="directory where the created experiment will be located")
    parser.add_argument("--id", required=True, help="prefix to name of the experiment/data reconstruction")
    parser.add_argument("--scan", required=True, help="a range of scans to prepare data from")
    parser.add_argument("--beamline", required=True, help="beamline")
    parser.add_argument("--data_dir", required=True, help="raw data directory")
    parser.add_argument("--darkfield_filename", required=True, help="dark field file name")
    parser.add_argument("--whitefield_filename", required=True, help="white field file name")
    parser.add_argument("--specfile", required=True, help="full name, including path of specfile")
    parser.add_argument("--diffractometer", required=True, help="diffractometer")

    args = parser.parse_args()

    return create_exp(args.working_dir, args.id, args.scan, args.beamline, args.data_dir, args.darkfield_filename,
                      args.whitefield_filename, args.specfile, args.diffractometer)


if __name__ == "__main__":
    exit(main())

# python create_experiment.py --working_dir example_workspace --id s2 --scan 54 --beamline aps_34idc --data_dir example_data/AD34idcTIM2_example --darkfield_filename example_data/dark.tif --whitefield_filename example_data/whitefield.tif --specfile example_data/example.spec --diffractometer 34idc

