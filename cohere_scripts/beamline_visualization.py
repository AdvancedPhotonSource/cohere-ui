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
import inner_scripts.multipeak as mp
import inner_scripts.common as com
import inner_scripts.postprocess_utils as pu


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
    all_params = {k:v for d in config_maps.values() for k,v in d.items()}
    [res_dir, scan] = res_dir_scan

    save_dir = res_dir.replace('_phasing', '_viz')
    # create dir if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    beamline = all_params["beamline"]
    try:
        instr_module = importlib.import_module(f'beamlines.{beamline}.instrument')
    except Exception as e:
        print(e)
        print(f'cannot import beamlines.{beamline}.instrument module.')
        return (f'cannot import beamlines.{beamline}.instrument module.')

    instr_obj = instr_module.create_instr(all_params)
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
    (res_viz_d, res_viz_r) = (None, None)

    supportfile = ut.join(res_dir, 'support.npy')
    if os.path.isfile(supportfile):
        try:
            support = np.load(supportfile)
            # ut.save_tif(support, ut.join(save_dir, 'support.tif'))
        except:
            print(f'cannot load file {supportfile}')
    else:
        print(f'support file is missing in {res_dir} directory')

    # get geometry
    geometry = instr_obj.get_geometry(image.shape, scan, all_params)

    # This block of code creates vts file with arrays depending on the complex_mode.
    # If complex_mode is "AmpPhase" the following arrays are included: imAmp, imPh, support.
    # If complex_mode is "ReIm" the following arrays are included: imRe, imImag, support .
    dir_viz = pu.make_image_viz(geometry, image, support, all_params)
    complex_mode = all_params.get('complex_mode', 'AmpPhase')
    filename = ut.join(save_dir, f'direct_space_images_{complex_mode}.vts')
    dir_viz.write(filename, complex_mode=complex_mode)
    del dir_viz

    # If 'interpolation_mode' and 'interpolation_resolution' parameters are configured then
    # the image is interpolated. First the resolution is determined. It can be configured
    # or calculated depending on the 'interpolation_resolution' parameter.
    if 'interpolation_mode' in all_params and 'interpolation_resolution' in all_params:
        # Find the resolution, as it can be configured as a value or prompted to derive it if configured
        # to 'min-deconv_res'.
        match all_params['interpolation_resolution']:
            case [*_]:
                interpolation_resolution = all_params['interpolation_resolution']
            case int():
                interpolation_resolution = all_params['interpolation_resolution']
            case 'min-deconv_res':
                # Only direct resolution is needed for interpolation.
                # If configured to determine resolution, get it here in direct and reciprocal spaces, otherwise
                # get only direct space resolution to use for interpolation.
                only_direct_res = 'determine_resolution' not in all_params
                res_viz_d, res_viz_r = pu.make_resolution_viz(geometry, np.abs(image), all_params, only_direct=only_direct_res)
                res_ssg = res_viz_d.get_structured_grid()
                res_arr = res_ssg.point_data['resolution']
                # because of [::-1] to get array indexing right the x axis in paraview is last axis in array.
                res_arr.shape = res_ssg.dimensions[::-1]
                res_bounds = pu.find_datarange(res_arr, 0, 0.5)
                resd = [x / y for x, y in zip(res_bounds, res_arr.shape)]
                r1 = np.dot(geometry[1], [res_bounds[0] / res_arr.shape[0], 0, 0])
                r2 = np.dot(geometry[1], [0, res_bounds[1] / res_arr.shape[1], 0])
                r3 = np.dot(geometry[1], [0, 0, res_bounds[2] / res_arr.shape[2]])
                # interpolate at half the smallest value.  Could make a param.
                interpolation_resolution = min([np.linalg.norm(r1), np.linalg.norm(r2), np.linalg.norm(r3)]) / 2

        dir_viz = pu.make_image_viz(geometry, image, support, all_params)
        interpolation_mode = all_params['interpolation_mode']
        interpolated_data = pu.get_interpolated_arrays(dir_viz, interpolation_resolution, interpolation_mode=interpolation_mode)
        filename = ut.join(save_dir, f'direct_space_images_interpolated_{interpolation_mode}.vti')
        match interpolation_mode:
            case 'AmpPhase':
                # In this mode the image amplitudes and phases are obtained from the grid and interpolated
                # The imAmp and imPh arrays are then saved.
                pass
            case 'Complex':
                # In this mode the image amplitudes and phases are calculated from real and imaginary values
                # obtained from the grid and then interpolated.
                # The imAmp and imPh arrays are then saved.
                interpolated_data.point_data['imAmp'] = np.abs(interpolated_data.point_data['imRe'] +
                                                               1j * interpolated_data.point_data['imImag'])
                interpolated_data.point_data['imPh'] = np.angle(interpolated_data.point_data['imRe'] +
                                                               1j * interpolated_data.point_data['imImag'])
        interpolated_data.save(filename)

        del dir_viz

    if 'determine_resolution' in all_params:
        if res_viz_d is None:
            (res_viz_d, res_viz_r) = pu.make_resolution_viz(geometry, np.abs(image), all_params)
        res_viz_d.write(ut.join(save_dir, "resolution_direct.vts"))
        res_viz_r.write(ut.join(save_dir, "resolution_recip.vts"))
        del res_viz_d
        del res_viz_r

    if all_params.get('write_recip', False):
        #
        dfile = ut.join(*(os.path.split(res_dir)[0], "phasing_data", "data.tif"))
        d = ut.read_tif(dfile)
        ftim = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(image), norm='forward'))
        dviz = pu.make_recip_viz(geometry, np.abs(d), ftim)
        dviz.write(ut.join(save_dir, "reciprocal_space.vts"))

    if all_params.get('make_twin', False):
        twin_image = np.conjugate(np.flip(image))
        if support is not None:
            twin_support = np.flip(support)
        twin_viz = pu.make_image_viz(geometry, twin_image, twin_support)
        twin_viz.write(ut.join(save_dir, "twin_direct_space_images.vts"))

    cohfile = ut.join(res_dir, 'coherence.npy')
    if os.path.isfile(cohfile):
        try:
            coh = np.load(cohfile)
            (viz_d, viz_r) = pu.make_coherence_viz(geometry, coh, image.shape)
            viz_d.write(ut.join(save_dir, "direct_space_coherence.vts"))
            viz_r.write(ut.join(save_dir, "recip_space_coherence.vts"))
        except:
            raise


def handle_visualization(experiment_dir, rec_id=None, **kwargs):
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
    print('starting visualization process')

    conf_list = ['config_disp', 'config_instr', 'config_data', 'config_rec']
    err_msg, conf_maps, converted = com.get_config_maps(experiment_dir, conf_list, **kwargs)
    if len(err_msg) > 0:
        return err_msg

    main_conf_map = conf_maps['config']

    if 'multipeak' in main_conf_map and main_conf_map['multipeak']:
        mp.process_dir(experiment_dir, make_twin=False)
        return

    separate = main_conf_map.get('separate_scans', False) or main_conf_map.get('separate_scan_ranges', False)

    if 'results_dir' in conf_maps['config_disp']:
        results_dir = conf_maps['config_disp']['results_dir'].replace(os.sep, '/')
        if not os.path.isdir(results_dir):
            print(f'the configured results_dir: {results_dir} does not exist')
            return (f'the configured results_dir: {results_dir} does not exist')
    elif separate or conf_maps['config_rec'].get('reconstructions', 1) > 1:
        results_dir = experiment_dir
    elif rec_id is not None:
        results_dir = ut.join(experiment_dir, f'results_phasing_{rec_id}')
    else:
        results_dir = ut.join(experiment_dir, 'results_phasing')
    # find directories with image.npy file in the root of results_dir
    dirs = []
    for (dirpath, dirnames, filenames) in os.walk(results_dir):
        for file in filenames:
            if file.endswith('image.npy'):
                dirs.append((dirpath).replace(os.sep, '/'))
    if len(dirs) == 0:
        print(f'no image.npy files found in the directory tree {results_dir}')
        return (f'no image.npy files found in the directory tree {results_dir}')

    if separate:
        scans = []
        # the scan that will be used to derive geometry is determined from the scan directory
        for dir in dirs:
            subdir = dir.removeprefix(f'{experiment_dir}/')
            if subdir.startswith('scan'):
                scan_dir = subdir.split('/')[0]
                scans.append(int(scan_dir.removeprefix('scan_').split('-')[-1]))
            else:
                print(f'directory {dir} does not start with "scan", not visualizing')
        dirs = list(zip(dirs, scans))
    else:
        last_scan = int(main_conf_map['scan'].split(',')[-1].split('-')[-1])
        dirs = [[dir, last_scan] for dir in dirs]

    if len(dirs) == 1:
        process_dir(conf_maps, dirs[0])
    else:
        func = partial(process_dir, conf_maps)
        no_proc = min(cpu_count(), len(dirs))
        with Pool(processes=no_proc) as pool:
            pool.map_async(func, dirs)
            pool.close()
            pool.join()
    print('done with post processing for visualization')
    return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory")
    parser.add_argument("--rec_id", help="alternate reconstruction id")
    parser.add_argument("--debug", action="store_true",
                        help="if True the vrifier has no effect on processing")
    args = parser.parse_args()
    handle_visualization(args.experiment_dir, args.rec_id, debug=args.debug)


if __name__ == "__main__":
    main()

# python run_disp.py experiment_dir
# main calls handle_viz
# handle viz calls process dir in pool
# process dir loads image, support, makes a CXDViz and calls vizualize method
# vizualize method.  calls write_ds_structuredgrid with array lists.
