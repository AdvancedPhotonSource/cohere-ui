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
           'get_conf_dict',
           'handle_visualization',
           'main']

import argparse
import sys
import os
import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
import importlib
import convertconfig as conv
import cohere
import util.util as ut
from tvtk.api import tvtk


class CXDViz:
    """
    CXDViz(self, crop, geometry)
    ===================================

    Class, generates files for visualization from reconstructed suite.

    crop : list
        list of fractions; the fractions will be multipled by dimensions to derive region to visualize
    geometry : tuple of arrays
        arrays containing geometry in reciprocal and direct space

    """
    __all__ = ['visualize'
               ]

    dir_arrs = {}
    recip_arrs = {}

    def __init__(self, crop, geometry):
        """
        The constructor creates objects assisting with visualization.
        Parameters
        ----------
        crop : tuple or list
            list of fractions; the fractions will be applied to each dimension to derive region to visualize
        geometry : tuple of arrays
            arrays containing geometry in reciprocal and direct space
        Returns
        -------
        constructed object
        """
        self.crop = crop
        self.Trecip, self.Tdir = geometry
        self.dirspace_uptodate = 0
        self.recipspace_uptodate = 0


    def visualize(self, image, support, coh, save_dir, is_twin=False):
        """
        Manages visualization process. Saves the results in a given directory in files: image.vts, support.vts, and coherence.vts. If is_twin then the saved files have twin prefix.

        Parameters
        ----------
        image : ndarray
            image array
        support : ndarray
            support array or None
        coh : ndarray
            coherence array or None
        save_dir : str
            a directory to save the results
        is_twin : boolean
            True if the image array is result of reconstruction, False if is_twin of reconstructed array.

        """
        save_dir = save_dir.replace(os.sep, '/')
        arrays = {"imAmp": abs(image), "imPh": np.angle(image)}
        self.add_ds_arrays(arrays)
        if is_twin:
            self.write_directspace(save_dir + '/twin_image')
        else:
            self.write_directspace(save_dir + '/image')
        self.clear_direct_arrays()
        if support is not None:
            arrays = {"support": support}
            self.add_ds_arrays(arrays)
            if is_twin:
                self.write_directspace(save_dir + '/twin_support')
            else:
                self.write_directspace(save_dir + '/support')
            self.clear_direct_arrays()

        if coh is not None:
            coh = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(coh)))
            coh = ut.get_zero_padded_centered(coh, image.shape)
            arrays = {"cohAmp": np.abs(coh), "cohPh": np.angle(coh)}
            self.add_ds_arrays(arrays)
            self.write_directspace(save_dir + '/coherence')
            self.clear_direct_arrays()


    def update_dirspace(self, shape):
        """
        Updates direct space grid.
        Parameters
        ----------
        shape : tuple
            shape of reconstructed array
        Returns
        -------
        nothing
        """
        dims = list(shape)
        self.dxdir = 1.0 / shape[0]
        self.dydir = 1.0 / shape[1]
        self.dzdir = 1.0 / shape[2]

        r = np.mgrid[
            0:dims[0] * self.dxdir:self.dxdir, \
            0:dims[1] * self.dydir:self.dydir, \
            0:dims[2] * self.dzdir:self.dzdir]

        r.shape = 3, dims[0] * dims[1] * dims[2]

        self.dir_coords = np.dot(self.Tdir, r).transpose()

        self.dirspace_uptodate = 1

    def update_recipspace(self, shape):
        """
        Updates reciprocal space grid.
        Parameters
        ----------
        shape : tuple
            shape of reconstructed array
        Returns
        -------
        nothing
        """
        dims = list(shape)
        q = np.mgrid[0:dims[0], 0:dims[1], 0:dims[2]]

        q.shape = 3, dims[0] * dims[1] * dims[2]

        self.recip_coords = np.dot(self.Trecip, q).transpose()
        self.recipspace_uptodate = 1


    def clear_direct_arrays(self):
        self.dir_arrs.clear()


    def clear_recip_arrays(self):
        self.recip_arrs.clear()


    def add_ds_arrays(self, named_arrays, logentry=None):
        names = sorted(list(named_arrays.keys()))
        shape = named_arrays[names[0]].shape
        if not self.are_same_shapes(named_arrays, shape):
            print('arrays in set should have the same shape')
            return
        # find crop beginning and ending
        [(x1, x2), (y1, y2), (z1, z2)] = self.get_crop_points(shape)
        for name in named_arrays.keys():
            self.dir_arrs[name] = named_arrays[name][x1:x2, y1:y2, z1:z2]
        if (not self.dirspace_uptodate):
            self.update_dirspace((x2 - x1, y2 - y1, z2 - z1))


    def are_same_shapes(self, arrays, shape):
        for name in arrays.keys():
            arr_shape = arrays[name].shape
            for i in range(len(shape)):
                if arr_shape[i] != shape[i]:
                    return False
        return True


    def get_crop_points(self, shape):
        # shape and crop should be 3 long
        crop_points = []
        for i in range(len(shape)):
            cropped_size = int(shape[i] * self.crop[i])
            chopped = int((shape[i] - cropped_size) / 2)
            crop_points.append((chopped, chopped + cropped_size))
        return crop_points


    def get_ds_structured_grid(self, **args):
        sg = tvtk.StructuredGrid()
        arr0 = self.dir_arrs[list(self.dir_arrs.keys())[0]]
        dims = list(arr0.shape)
        sg.points = self.dir_coords
        for a in self.dir_arrs.keys():
            arr = tvtk.DoubleArray()
            arr.from_array(self.dir_arrs[a].ravel())
            arr.name = a
            sg.point_data.add_array(arr)

        sg.dimensions = (dims[2], dims[1], dims[0])
        sg.extent = 0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1
        return sg


    def get_rs_structured_grid(self, **args):
        sg = tvtk.StructuredGrid()
        arr0 = self.recip_arrs[list(self.recip_arrs.keys())[0]]
        dims = list(arr0.shape)
        sg.points = self.recip_coords
        for a in self.recip_arrs.keys():
            arr = tvtk.DoubleArray()
            arr.from_array(self.recip_arrs[a].ravel())
            arr.name = a
            sg.point_data.add_array(arr)

        sg.dimensions = (dims[2], dims[1], dims[0])
        sg.extent = 0, dims[2] - 1, 0, dims[1] - 1, 0, dims[0] - 1
        return sg


    def write_directspace(self, filename, **args):
        filename = filename.replace(os.sep, '/')
        sgwriter = tvtk.XMLStructuredGridWriter()
        # sgwriter.file_type = 'binary'
        if filename.endswith(".vtk"):
            sgwriter.file_name = filename
        else:
            sgwriter.file_name = filename + '.vts'
        sgwriter.set_input_data(self.get_ds_structured_grid())
        sgwriter.write()
        print('saved file', filename)


    def write_recipspace(self, filename, **args):
        filename = filename.replace(os.sep, '/')
        sgwriter = tvtk.XMLStructuredGridWriter()
        if filename.endswith(".vtk"):
            sgwriter.file_name = filename
        else:
            sgwriter.file_name = filename + '.vts'
        sgwriter.set_input_data(self.get_rs_structured_grid())
        sgwriter.write()
        print('saved file', filename)


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
    imagefile = res_dir + '/image.npy'
    try:
        image = np.load(imagefile)
        cohere.save_tif(image, save_dir + '/image.tif')
    except:
        print('cannot load file', imagefile)
        return

    support = None
    coh = None

    supportfile = res_dir + '/support.npy'
    if os.path.isfile(supportfile):
        try:
            support = np.load(supportfile)
            cohere.save_tif(support, save_dir + '/support.tif')
        except:
            print('cannot load file', supportfile)
    else:
        print('support file is missing in ' + res_dir + ' directory')

    cohfile = res_dir + '/coherence.npy'
    if os.path.isfile(cohfile):
        try:
            coh = np.load(cohfile)
        except:
            print('cannot load file', cohfile)

    if support is not None:
        image, support = ut.center(image, support)
    if rampups > 1:
        image = ut.remove_ramp(image, ups=rampups)

    viz = CXDViz(crop, geometry)
    viz.visualize(image, support, coh, save_dir)

    if make_twin:
        image = np.conjugate(np.flip(image))
        if support is not None:
            support = np.flip(support)
            image, support = ut.center(image, support)
        if rampups > 1:
            image = ut.remove_ramp(image, ups=rampups)
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
    image_file = image_file.replace(os.sep, '/')
    if os.path.isfile(image_file):
        try:
            image = np.load(image_file)
        except:
            print('cannot load file', image_file)
    else:
        print(image_file, 'file is missing')
        return

    if rampups > 1:
        image = ut.remove_ramp(image, ups=rampups)

    viz = CXDViz(crop, geometry)
    viz.visualize(image, None, None, os.path.dirname(image_file).replace(os.sep, '/'))


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
    experiment_dir = experiment_dir.replace(os.sep, '/')
    if not os.path.isdir(experiment_dir):
        print("Please provide a valid experiment directory")
        return None
    conf_dir = experiment_dir + '/conf'

    main_conf_file = conf_dir + '/config'
    main_conf_map = ut.read_config(main_conf_file)
    if main_conf_map is None:
        return None

    # convert configuration files if needed
    if 'converter_ver' not in main_conf_map or conv.get_version() is None or conv.get_version() < main_conf_map[
        'converter_ver']:
        conv.convert(conf_dir)
        # re-parse config
        main_conf_map = ut.read_config(main_conf_file)

    er_msg = cohere.verify('config', main_conf_map)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return None

    disp_conf = conf_dir + '/config_disp'

    # parse the conf once here and save it in dictionary, it will apply to all images in the directory tree
    conf_dict = ut.read_config(disp_conf)
    if conf_dict is None:
        return None
    er_msg = cohere.verify('config_disp', conf_dict)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        return None

    if 'beamline' in main_conf_map:
        conf_dict['beamline'] = main_conf_map['beamline']
    else:
        print('Beamline must be configured in configuration file ' + main_conf_file)
        return None

    # get specfile and last_scan from the config file and add it to conf_dict
    if 'specfile' in main_conf_map and 'scan' in main_conf_map:
        conf_dict['specfile'] = main_conf_map['specfile'].replace(os.sep, '/')
        scan = main_conf_map['scan']
        last_scan = scan.split(',')[-1].split('-')[-1]
        conf_dict['last_scan'] = int(last_scan)
    else:
        print("specfile or scan range not in main config")

    # get binning from the config_data file and add it to conf_dict
    data_conf = conf_dir + '/config_data'
    data_conf_map = ut.read_config(data_conf)
    if data_conf_map is None:
        return conf_dict
    if 'binning' in data_conf_map:
        conf_dict['binning'] = data_conf_map['binning']
    if 'separate_scans' in data_conf_map and data_conf['separate_scans'] or 'separate_scan_ranges' in data_conf_map and  data_conf['separate_scan_ranges']:
        conf_dict['separate'] = True
    else:
        conf_dict['separate'] = False

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
    experiment_dir = experiment_dir.replace(os.sep, '/')
    print ('starting visualization process')
    conf_dict = get_conf_dict(experiment_dir)
    if conf_dict is None:
        return

    try:
        disp = importlib.import_module('beamlines.' + conf_dict['beamline'] + '.disp')
    except:
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
        make_twin = False

    if image_file is not None:
        # find shape without loading the array
        with open(image_file, 'rb') as f:
            np.lib.format.read_magic(f)
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        geometry = disp.set_geometry(shape, params)
        process_file(image_file, geometry, rampups, params.crop)
        return
    elif conf_dict['separate']:
        results_dir = experiment_dir
    elif rec_id is not None:
        results_dir = experiment_dir + '/results_phasing_' + rec_id
    else:
        if 'results_dir' in conf_dict:
            results_dir = conf_dict['results_dir'].replace(os.sep, '/')
        else:
            results_dir = experiment_dir
    # find directories with image.npy file in the root of results_dir
    dirs = []
    for (dirpath, dirnames, filenames) in os.walk(results_dir):
        for file in filenames:
            if file.endswith('image.npy'):
                dirs.append((dirpath).replace(os.sep, '/'))
    if len(dirs) == 0:
        print ('no image.npy files found in the directory tree', results_dir)
        return
    else:
        # find shape without loading the array
        with open(dirs[0] + '/image.npy', 'rb') as f:
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
