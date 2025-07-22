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
import cohere_core.utilities as ut
import cohere_ui.api.multipeak as mp
import cohere_ui.api.common as com
import cohere_ui.api.postprocess_utils as pu


def process_dir(config_maps, res_dir_scan):
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
    [scan, res_dir] = res_dir_scan

    save_dir = res_dir.replace('_phasing', '_viz')
    # create dir if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    config_map = config_maps['config']
    beamline = config_map["beamline"]
    try:
        instr_module = importlib.import_module(f'cohere_ui.beamlines.{beamline}.instrument')
    except Exception as e:
        print(e)
        print(f'cannot import cohere_ui.beamlines.{beamline}.instrument module.')
        return (f'cannot import cohere_ui.beamlines.{beamline}.instrument module.')

    instr_obj = instr_module.create_instr(config_maps)
    if instr_obj is None:
        print ('cannot create instrument, check configuration, exiting')
        return

    # image file was checked in calling function
    imagefile = ut.join(res_dir, 'image.npy')
    try:
        image = np.load(imagefile)
        # ut.save_tif(image, ut.join(save_dir, 'support.tif'))
    except:
        print(f'cannot load file {imagefile}')
        return

    # init variables
    support = None
    coh = None
    #(res_viz_d, res_viz_r) = (None, None)

    supportfile = ut.join(res_dir, 'support.npy')
    if os.path.isfile(supportfile):
        try:
            support = np.load(supportfile)
            # ut.save_tif(support, ut.join(save_dir, 'support.tif'))
        except:
            print(f'cannot load file {supportfile}')
    else:
        print(f'support file is missing in {res_dir} directory')

    viz_params = config_maps['config_disp']
    # get geometry
    geometry = instr_obj.get_geometry(image.shape, scan, config_maps)

    # This block of code creates vts file with arrays depending on the complex_mode.
    # If complex_mode is "AmpPhase" the following arrays are included: imAmp, imPh, support.
    # If complex_mode is "ReIm" the following arrays are included: imRe, imImag, support .
    dir_viz = pu.make_image_viz(geometry, image, support,config_maps)
    complex_mode = viz_params.get('complex_mode', 'AmpPhase')
    filename = ut.join(save_dir, f'direct_space_images_{complex_mode}.vts')
    dir_viz.write(filename, complex_mode=complex_mode)
    print(f'saved direct_space_images_{complex_mode}.vts file')

    res_viz_d, res_viz_r = None, None
    # If 'interpolation_mode' and 'interpolation_resolution' parameters are configured then
    # the image is interpolated. First the resolution is determined. It can be configured
    # or calculated depending on the 'interpolation_resolution' parameter.
    if 'interpolation_mode' in viz_params:
        if 'interpolation_resolution' not in viz_params:
            print(f'interpolation_resolution parameter not configured, exiting')
            return

        # Find the resolution, as it can be configured as a value or prompted to derive it if configured
        # to 'min_deconv_res'.
        match viz_params['interpolation_resolution']:
            case [*_]:
                interpolation_resolution = viz_params['interpolation_resolution']
            case int():
                interpolation_resolution = viz_params['interpolation_resolution']
            case float():
                interpolation_resolution = viz_params['interpolation_resolution']
            case 'min_deconv_res':
                # Only direct resolution is needed for interpolation.
                # If configured to determine resolution, get it here in direct and reciprocal spaces, otherwise
                # get only direct space resolution to use for interpolation.
                only_direct_res = 'determine_resolution' not in viz_params
                res_viz_d, res_viz_r = pu.make_resolution_viz(geometry, np.abs(image), config_maps)
                res_viz_d.write(ut.join(save_dir, "resolution_direct.vts"))
                res_viz_r.write(ut.join(save_dir, "resolution_recip.vts"))
                print('saved resolution_direct.vts and resolution_recip.vts files')
                res_ssg = res_viz_d.get_structured_grid()
                res_arr = res_ssg.point_data['resolution']
                # because of [::-1] to get array indexing right the x axis in paraview is last axis in array.
                res_arr.shape = res_ssg.dimensions[::-1]
                res_bounds = pu.find_datarange(res_arr, 0, 0.5)
                r1 = np.dot(geometry[1], [res_bounds[0] / res_arr.shape[0], 0, 0])
                r2 = np.dot(geometry[1], [0, res_bounds[1] / res_arr.shape[1], 0])
                r3 = np.dot(geometry[1], [0, 0, res_bounds[2] / res_arr.shape[2]])
                # interpolate at half the smallest value.  Could make a param.
                interpolation_resolution = min([np.linalg.norm(r1), np.linalg.norm(r2), np.linalg.norm(r3)]) / 2
            case _:
                print(f'not supported interpolation_resolution parameter value {viz_params["interpolation_resolution"]}, exiting')
                return

        interpolation_mode = viz_params['interpolation_mode']
        interpolated_data = pu.get_interpolated_arrays(dir_viz, interpolation_resolution, interpolation_mode=interpolation_mode)
        filename = ut.join(save_dir, f'direct_space_images_interpolated_{interpolation_mode}.vti')
        match interpolation_mode:
            case 'AmpPhase':
                # In this mode the image amplitudes and phases are obtained from the grid and interpolated
                # The imAmp and imPh arrays are then saved.
                pass
            case 'ReIm':
                # In this mode the image amplitudes and phases are calculated from real and imaginary values
                # obtained from the grid and then interpolated.
                # The imAmp and imPh arrays are then saved.
                interpolated_data.point_data['imAmp'] = np.abs(interpolated_data.point_data['imRe'] +
                                                               1j * interpolated_data.point_data['imImag'])
                interpolated_data.point_data['imPh'] = np.angle(interpolated_data.point_data['imRe'] +
                                                               1j * interpolated_data.point_data['imImag'])
            case _:
                print(f'not supported interpolation_mode parameter value {viz_params["interpolation_mode"]}, exiting')
                return

        interpolated_data.save(filename)
        print(f'saved direct_space_images_interpolated_{interpolation_mode}.vti file')

        del dir_viz

    if 'determine_resolution' in viz_params:
        if res_viz_d is None: # otherwise it was saved during interpolation
            res_viz_d, res_viz_r = pu.make_resolution_viz(geometry, np.abs(image), config_maps)
            res_viz_d.write(ut.join(save_dir, "resolution_direct.vts"))
            res_viz_r.write(ut.join(save_dir, "resolution_recip.vts"))
            print('saved resolution_direct.vts and resolution_recip.vts files')

        del res_viz_d
        del res_viz_r

    if viz_params.get('write_recip', False):
        dfile = ut.join(*(os.path.split(res_dir)[0], "phasing_data", "data.tif"))
        d = ut.read_tif(dfile)
        ftim = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(image), norm='forward'))
        dviz = pu.make_recip_viz(geometry, np.abs(d), ftim)
        dviz.write(ut.join(save_dir, "reciprocal_space.vts"))
        print('saved reciprocal_space.vts file')

    if viz_params.get('make_twin', False):
        twin_image = np.conjugate(np.flip(image))
        if support is not None:
            twin_support = np.flip(support)
        twin_viz = pu.make_image_viz(geometry, twin_image, twin_support, config_maps)
        twin_viz.write(ut.join(save_dir, "twin_direct_space_images.vts"))
        print('saved twin_direct_space_images.vts file')

    cohfile = ut.join(res_dir, 'coherence.npy')
    if os.path.isfile(cohfile):
        try:
            coh = np.load(cohfile)
            (viz_d, viz_r) = pu.make_coherence_viz(geometry, coh, image.shape)
            viz_d.write(ut.join(save_dir, "direct_space_coherence.vts"))
            viz_r.write(ut.join(save_dir, "recip_space_coherence.vts"))
            prinr('saved direct_space_coherence.vts and recip_space_coherence.vts files')
        except:
            raise


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

    conf_list = ['config_disp', 'config_instr', 'config_data', 'config_mp']
    conf_maps, converted = com.get_config_maps(experiment_dir, conf_list, **kwargs)

    if 'config_disp' not in conf_maps.keys():
        print('missing config_disp file')
    if 'config_instr' not in conf_maps.keys():
        return 'missing config_instr file, exiting'
    if 'config_data' not in conf_maps.keys():
        print('no config_data file')

    main_conf_map = conf_maps['config']

    if 'multipeak' in main_conf_map and main_conf_map['multipeak']:
        mp.process_dir(experiment_dir, conf_maps)
    else:
        separate = main_conf_map.get('separate_scans', False) or main_conf_map.get('separate_scan_ranges', False)
        rec_id = kwargs.get('rec_id', None)
        if 'results_dir' in conf_maps:
            results_dir = conf_maps['config_disp']['results_dir'].replace(os.sep, '/')
            if rec_id is not None and not results_dir.endswith(rec_id):
                print(f'Verify the results_directory. Currently set to {results_dir}')
            if separate and results_dir != experiment_dir:
                print(f'Verify the results_directory. Currently set to {results_dir}')
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
            process_dir(conf_maps, scans_dirs[0])
        else:
            func = partial(process_dir, conf_maps)
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