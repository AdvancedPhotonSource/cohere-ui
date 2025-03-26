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
           'handle_visualization',
           'main']

import argparse
import os
import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
import importlib
import traceback
import cohere_core.utilities as ut
from tvtk.api import tvtk
import inner_scripts.multipeak as mp
import inner_scripts.common as com


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
    __all__ = ['visualize']

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


    def visualize(self, image, support, coh, save_dir, unwrap=False, is_twin=False):
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
        unwrap : boolean
            If True it will unwrap phase
        is_twin : boolean
            True if the image array is result of reconstruction, False if is_twin of reconstructed array.
        """
        save_dir = save_dir.replace(os.sep, '/')
        arrays = {"imAmp": abs(image), "imPh": np.angle(image)}

        # unwrap phase here
        if unwrap:
            from skimage import restoration
            arrays['imUwPh'] = restoration.unwrap_phase(arrays['imPh'])

        self.add_ds_arrays(arrays)
        if is_twin:
            self.write_directspace(ut.join(save_dir, 'twin_image'))
        else:
            self.write_directspace(ut.join(save_dir, 'image'))
        self.clear_direct_arrays()
        if support is not None:
            arrays = {"support": support}
            self.add_ds_arrays(arrays)
            if is_twin:
                self.write_directspace(ut.join(save_dir, 'twin_support'))
            else:
                self.write_directspace(ut.join(save_dir, 'support'))
            self.clear_direct_arrays()

        if coh is not None:
            coh = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(coh)))
            coh = ut.pad_center(coh, image.shape)
            arrays = {"cohAmp": np.abs(coh), "cohPh": np.angle(coh)}
            self.add_ds_arrays(arrays)
            self.write_directspace(ut.join(save_dir, 'coherence'))
            self.clear_direct_arrays()


    def update_dirspace(self, shape, orig_shape):
        """
        Updates direct space grid.
        Parameters
        ----------
        shape : tuple
            shape of reconstructed array
        orig_shape : tuple
            shape of array before binning
        Returns
        -------
        nothing
        """
        dims = list(shape)
        self.dxdir = 1.0 / orig_shape[0]
        self.dydir = 1.0 / orig_shape[1]
        self.dzdir = 1.0 / orig_shape[2]

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
            self.update_dirspace((x2 - x1, y2 - y1, z2 - z1), shape)


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
        print(f'saved file {filename}')


    def write_recipspace(self, filename, **args):
        filename = filename.replace(os.sep, '/')
        sgwriter = tvtk.XMLStructuredGridWriter()
        if filename.endswith(".vtk"):
            sgwriter.file_name = filename
        else:
            sgwriter.file_name = f'{filename}.vts'
        sgwriter.set_input_data(self.get_rs_structured_grid())
        sgwriter.write()
        print(f'saved file {filename}')

def voxel_size(geometry, shape):
    (Tr, Td)  = geometry
    print('geometry', Tr, Td)
    B_recip = np.stack([Tr[1, :], Tr[0, :], Tr[2, :]])
    print('B_recip', B_recip)
    rs_voxel_size = np.max([np.linalg.norm(B_recip[:, i]) for i in range(3)])  # Units are inverse nanometers
    print('rs voxel size', rs_voxel_size)
    ds_voxel_size = 2*np.pi/(rs_voxel_size*shape[0])
    print('ds voxel size', ds_voxel_size)


def process_dir(instr_conf_map, config_map, res_scans_dirs):
    """
    Creates and saves file in vts format that represents the phasing results found in the giving directory applying
    the parameters from configuration files.

    :param instr_conf_map:
    :param config_map:
    :param res_scans_dirs: list
        contain two elements:
        scan - scan (last scan) relative to the res_dir
        res-dir - directory where the results of reconstruction are saved and will be processes
    :return:
    """
    [scan, res_dir] = res_scans_dirs
    if 'save_dir' in config_map:
        save_dir = config_map['save_dir']
    else:
        save_dir = res_dir.replace('_phasing', '_viz')
    # create dir if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # image file was checked in calling function
    imagefile = ut.join(res_dir, 'image.npy')
    try:
        image = np.load(imagefile)
        ut.save_tif(image, ut.join(save_dir, 'image.tif'))
    except:
        print(f'cannot load file {imagefile}')
        return
    shape = image.shape

    # init support and coh, will be overridden if not None
    support = None
    coh = None

    supportfile = ut.join(res_dir, 'support.npy')
    if os.path.isfile(supportfile):
        try:
            support = np.load(supportfile)
            ut.save_tif(support, ut.join(save_dir, 'support.tif'))
        except:
            print(f'cannot load file {supportfile}')
    else:
        print(f'support file is missing in {res_dir} directory')

    beamline = config_map["beamline"]
    try:
        instr_module = importlib.import_module(f'beamlines.{beamline}.instrument')
    except Exception as e:
        print(e)
        print(f'cannot import beamlines.{beamline}.instrument module.')
        return (f'cannot import beamlines.{beamline}.instrument module.')

    instr_obj = instr_module.create_instr(instr_conf_map)
    geometry = None
    try:
        geometry = instr_obj.get_geometry(shape, scan, instr_conf_map)
    except:
        raise

    cohfile = ut.join(res_dir, 'coherence.npy')
    if os.path.isfile(cohfile):
        try:
            coh = np.load(cohfile)
        except:
            print(f'cannot load file {cohfile}')

    if config_map.get('rampups', 1) > 1:
        import cohere_core.utilities.dvc_utils as dvut

        dvut.set_lib_from_pkg('np')
        rampups = config_map.get('rampups', 1)
        image = dvut.remove_ramp(image, ups=rampups)

    unwrap = config_map.get('unwrap', False)
    crop = config_map.get('crop', [1., 1., 1.])
    crop = crop + [1.0] * (len(image.shape) - len(crop))
    viz = CXDViz(crop, geometry)
    viz.visualize(image, support, coh, save_dir, unwrap)

    if config_map.get('make_twin', False):
        image = np.conjugate(np.flip(image))
        if support is not None:
            support = np.flip(support)
        viz.visualize(image, support, coh, save_dir, unwrap, True)


def handle_visualization(experiment_dir, **kwargs):
    """
    If the image_file parameter is defined, the file is processed and vts file saved. Otherwise this function determines root directory with results that should be processed for visualization. Multiple images will be processed concurrently.
    Parameters
    ----------
    experiment_dir : str
        directory where the experiment files are saved
    kwargs: ver parameters
        may contain:
        - rec_id : reconstruction id, pointing to alternate config
        - no_verify : boolean switch to determine if the verification error is returned
        - debug : boolean switch not used in this code
    Returns
    -------
    nothing
    """
    print ('starting visualization process')

    conf_list = ['config_disp', 'config_instr', 'config_data']
    err_msg, conf_maps, converted = com.get_config_maps(experiment_dir, conf_list, **kwargs)
    if len(err_msg) > 0:
        return err_msg
    # check the maps
    if 'config_disp' not in conf_maps.keys():
        print('missing config_disp file')
    if 'config_instr' not in conf_maps.keys():
        return 'missing config_instr file, exiting'
    if 'config_data' not in conf_maps.keys():
        print('no config_data file')

    main_conf_map = conf_maps['config']
    instr_conf_map = conf_maps['config_instr']
    data_conf_map = conf_maps.get('config_data', {})
    # add binning to instrument configuration, as this affects instrument processing
    instr_conf_map['binning'] = data_conf_map.get('binning', [1, 1, 1])
    disp_conf_map = conf_maps.get('config_disp', {})

    if 'multipeak' in main_conf_map and main_conf_map['multipeak']:
        mp.process_dir(experiment_dir, make_twin=False)
    else:
        separate = main_conf_map.get('separate_scans', False) or main_conf_map.get('separate_scan_ranges', False)
        # get parameters from config files
        conf_map = disp_conf_map
        conf_map['beamline'] = main_conf_map.get('beamline')

        rec_id = kwargs.get('rec_id', None)
        if 'results_dir' in disp_conf_map:
            results_dir = disp_conf_map['results_dir'].replace(os.sep, '/')
            if rec_id is not None and not results_dir.endswith(rec_id):
                print(f'Verify the results directory, Currently set to {results_dir}')
            if separate and results_dir != experiment_dir:
                print(f'Verify the results directory, Currently set to {results_dir}')
            if not os.path.isdir(results_dir):
                print(f'the configured results_dir: {results_dir} does not exist')
                return(f'the configured results_dir: {results_dir} does not exist')
        elif separate:
            results_dir = experiment_dir
        elif rec_id is not None:
            results_dir = ut.join(experiment_dir, f'results_phasing_{kwargs["rec_id"]}')
        else:
            results_dir = ut.join(experiment_dir, 'results_phasing')

        # find directories with image.npy file in the root of results_dir
        scandirs = []
        for (dirpath, dirnames, filenames) in os.walk(results_dir):
            for file in filenames:
                if file.endswith('image.npy'):
                    scandirs.append((dirpath).replace(os.sep, '/'))
        if len(scandirs) == 0:
            print (f'no image.npy files found in the directory tree {results_dir}')
            return (f'no image.npy files found in the directory tree {results_dir}')

        scans_dirs = []
        if separate:
            # the scan that will be used to derive geometry is determined from the scan directory
            # the code below finds the last scan
            for dir in scandirs:
                # go up dir until reaching scan dir
                scandir_path = dir.split('/')
                i = -1
                temp = scandir_path[i]
                while not temp.startswith('scan'):
                    i -= 1
                    temp = scandir_path[i]
                scan_subdir = temp
                scans_dirs.append((int(scan_subdir.split('_')[-1].split('-')[-1]), dir))
        else:
            last_scan = int(main_conf_map['scan'].split(',')[-1].split('-')[-1])
            scans_dirs = [[last_scan, dir] for dir in scandirs]

        if len(scans_dirs) == 1:
            process_dir(instr_conf_map, conf_map, scans_dirs[0])
        else:
            func = partial(process_dir, instr_conf_map, conf_map)
            # TODO account for available memory when calculating number of processes
            # Currently the code will hung if not enough memory
            # Work around is to lower no_proc
            no_proc = min(cpu_count(), len(scandirs))
            with Pool(processes = no_proc) as pool:
               pool.map_async(func, scans_dirs)
               pool.close()
               pool.join()
    print ('done with processing display')
    return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory")
    parser.add_argument("--rec_id", action="store", help="alternate reconstruction id")
    parser.add_argument("--no_verify", action="store_true",
                        help="if True the verifier has no effect on processing, error is always printed when incorrect configuration")
    parser.add_argument("--debug", action="store_true",
                        help="not used currently, available to developer for debugging")
    args = parser.parse_args()
    handle_visualization(args.experiment_dir, rec_id=args.rec_id, no_verify=args.no_verify, debug=args.debug)


if __name__ == "__main__":
    main()

# python run_disp.py experiment_dir
