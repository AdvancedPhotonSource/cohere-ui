# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script processes reconstructed image for visualization.

After the script is executed the experiment directory will contain image.vts file for each reconstructed image in the given directory tree.
"""

__author__ = "Ross Harder"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['process_dir',
           'save_vtk_file',
           'get_conf_dict',
           'handle_visualization',
           'main']

import cohere.utilities.viz_util as vu
import cohere.utilities.utils as ut
from cohere.beamlines.viz import CXDViz
# import config_verifier as ver
import argparse
import sys
import os
import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
import importlib
import convertconfig as conv


def process_dir(geometry, rampups, crop, make_twin, res_dir):
    """
    Loads arrays from files in results directory. If reciprocal array exists, it will save reciprocal info in tif format. It calls the save_CX function with the relevant parameters.

    Parameters
    ----------
    res_dir_conf : tuple
        tuple of two elements:
        res_dir - directory where the results of reconstruction are saved
        conf_dict - dictionary containing configuration parameters

    Returns
    -------
    nothing
    """
    save_dir = res_dir.replace('_phasing', '_viz')
    # create dir if does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # image file was checked in calling function
    imagefile = os.path.join(res_dir, 'image.npy')
    try:
        image = np.load(imagefile)
        ut.save_tif(image, os.path.join(save_dir, 'image.tif'))
    except:
        print('cannot load file', imagefile)
        return

    support = None
    coh = None

    supportfile = os.path.join(res_dir, 'support.npy')
    if os.path.isfile(supportfile):
        try:
            support = np.load(supportfile)
            ut.save_tif(support, os.path.join(save_dir, 'support.tif'))
        except:
            print('cannot load file', supportfile)
    else:
        print('support file is missing in ' + res_dir + ' directory')

    cohfile = os.path.join(res_dir, 'coherence.npy')
    if os.path.isfile(cohfile):
        try:
            coh = np.load(cohfile)
        except:
            print('cannot load file', cohfile)

    if support is not None:
        image, support = vu.center(image, support)
    if rampups > 1:
        image = vu.remove_ramp(image, ups=rampups)

    viz = CXDViz(crop, geometry)
    viz.visualize(image, support, coh, save_dir)

    if make_twin:
        image = np.conjugate(np.flip(image))
        if support is not None:
            support = np.flip(support)
            image, support = vu.center(image, support)
        if rampups > 1:
            image = vu.remove_ramp(image, ups=rampups)
        viz.visualize(image, support, coh, save_dir, True)


def process_file(image_file, geometry, rampups, crop):
    """
    Loads array from given image file. Determines the vts file name and calls savw_CX function to process this  file. The vts file will have the same name as image file, with different extension and will be saved in the same directory.

    Parameters
    ----------
    image_file : str
        name of file in npy format containing reconstructrd image
    conf_dir : str
        dictionary containing configuration parameters

    Returns
    -------
    nothing
    """
    if os.path.isfile(image_file):
        try:
            image = np.load(image_file)
        except:
            print('cannot load file', image_file)
    else:
        print(image_file, 'file is missing')
        return

    if rampups > 1:
        image = vu.remove_ramp(image, ups=rampups)

    viz = CXDViz(crop, geometry)
    viz.visualize(image, None, None, os.path.dirname(image_file))


def get_conf_dict(experiment_dir):
    """
    Reads configuration files and creates dictionary with parameters that are needed for visualization.

    Parameters
    ----------
    experiment_dir : str
        directory where the experiment files are located

    Returns
    -------
    conf_dict : dict
        a dictionary containing configuration parameters
    """
    if not os.path.isdir(experiment_dir):
        print("Please provide a valid experiment directory")
        return None
    conf_dir = os.path.join(experiment_dir, 'conf')

    # convert configuration files if needed
    main_conf = os.path.join(conf_dir, 'config')
    main_config_map = ut.read_config(main_conf)
    if main_config_map is None:
        print("info: missing " + main_conf + " configuration file")
        return None

    if 'converter_ver' not in main_config_map or conv.get_version() is None or conv.get_version() < main_config_map['converter_ver']:
        main_config_map = conv.get_conf_dict(main_conf, 'config')

    disp_conf = os.path.join(conf_dir, 'config_disp')
    # # verify configuration file
    # if not ver.ver_config_disp(conf):
    #     print ('incorrect configuration file ' + conf +', cannot parse')
    #     return None

    # parse the conf once here and save it in dictionary, it will apply to all images in the directory tree
    conf_dict = ut.read_config(disp_conf)
    if conf_dict is None:
        return None
    #
    # # get specfile and last_scan from the config file and add it to conf_dict
    # main_conf = os.path.join(conf_dir, 'config')
    # if os.path.isfile(main_conf):
    #     config_map = ut.read_config(main_conf)
    #     if config_map is None:
    #         print ("info: can't read " + conf + " configuration file")
    #         return None

    if 'beamline' in main_config_map:
        conf_dict['beamline'] = main_config_map['beamline']
    else:
        print('beamline must be defined in the configuration file', main_conf)
        return None

    if 'specfile' in main_config_map and 'scan' in main_config_map:
        conf_dict['specfile'] = main_config_map['specfile']
        scan = main_config_map['scan']
        last_scan = scan.split(',')[-1].split('-')[-1]
        conf_dict['last_scan'] = int(last_scan)
    else:
        print("specfile or scan range not in main config")

    # get binning from the config_data file and add it to conf_dict
    data_conf = os.path.join(conf_dir, 'config_data')
    data_conf_map = ut.read_config(data_conf)
    if data_conf_map is None:
        return conf_dict
    if 'binning' in data_conf_map:
        conf_dict['binning'] = data_conf_map['binning']

    return conf_dict


def handle_visualization(experiment_dir, rec_id=None, image_file=None):
    """
    If the image_file parameter is defined, the file is processed and vts file saved. Otherwise this function determines root directory with results that should be processed for visualization. Multiple images will be processed concurrently.

    Parameters
    ----------
    conf_dir : str
        directory where the file will be saved

    Returns
    -------
    nothing
    """
    print ('starting visualization process')
    conf_dict = get_conf_dict(experiment_dir)
    if conf_dict is None:
        return

    if 'beamline' in conf_dict:
        disp = importlib.import_module('beamlines.' + conf_dict['beamline'] + '.disp')
    else:
        print ('cannot import beamlines.' + conf_dict['beamline'] + '.disp module.')
        return

    try:
        params = disp.DispalyParams(conf_dict)
    except Exception as e:
        print ('exception', e)
        return

    det_obj = None
    diff_obj = None
    try:
        detector_name = params.detector
        try:
            det = importlib.import_module('beamlines.aps_34idc.detectors')
            try:
                det_obj = det.create_detector(detector_name)
            except:
                print('detector', detector_name, 'is not defined in beamlines detectors')
        except:
            print('problem importing detectors file from beamline module')
    except:
        pass
    try:
        diffractometer_name = params.diffractometer
        try:
            diff = importlib.import_module('beamlines.aps_34idc.diffractometers')
            try:
                diff_obj = diff.create_diffractometer(diffractometer_name)
            except:
                print ('diffractometer', diffractometer_name, 'is not defined in beamlines detectors')
        except:
             print('problem importing diffractometers file from beamline module')
    except:
        pass

    if not params.set_instruments(det_obj, diff_obj):
        return

    try:
        rampups = params.rampsup
    except:
        rampups = 1

    if 'make_twin' in conf_dict:
        make_twin = conf_dict['make_twin']
    else:
        make_twin = True

    if image_file is not None:
        # find shape without loading the array
        with open(image_file, 'rb') as f:
            np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        geometry = disp.set_geometry(shape, params)
        process_file(image_file, geometry, rampups, params.crop)
        return
    elif rec_id is not None:
        results_dir = os.path.join(experiment_dir, 'results_phasing_' + rec_id)
    else:
        if 'results_dir' in conf_dict:
            results_dir = conf_dict['results_dir']
        else:
            results_dir = experiment_dir
    # find directories with image.npy file in the root of results_dir
    dirs = []
    for (dirpath, dirnames, filenames) in os.walk(results_dir):
        for file in filenames:
            if file.endswith('image.npy'):
                dirs.append((dirpath))
    if len(dirs) == 0:
        print ('no image.npy files found in the directory tree', results_dir)
        return
    else:
        # find shape without loading the array
        with open(os.path.join(dirs[0], 'image.npy'), 'rb') as f:
            np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        geometry = disp.set_geometry(shape, params)

    if len(dirs) == 1:
        process_dir(geometry, rampups, params.crop, make_twin, dirs[0])
    elif len(dirs) >1:
        func = partial(process_dir, geometry, rampups, params.crop, make_twin)
        no_proc = min(cpu_count(), len(dirs))
        with Pool(processes = no_proc) as pool:
           pool.map_async(func, dirs)
           pool.close()
           pool.join()
    print ('done with processing display')


def main(arg):
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory")
    parser.add_argument("--image_file", help="a file in .npy format to be processed for visualization")
    parser.add_argument("--rec_id", help="alternate reconstruction id")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir
    rec_id = args.rec_id
    if args.image_file:
        handle_visualization(experiment_dir, args.rec_id, args.image_file)
    else:
        handle_visualization(experiment_dir, args.rec_id)


if __name__ == "__main__":
    main(sys.argv[1:])

# python run_disp.py experiment_dir
