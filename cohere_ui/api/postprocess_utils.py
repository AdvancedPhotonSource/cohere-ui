# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import numpy as np
import cohere_core.utilities.dvc_utils as dvut
import cohere_core.utilities as ut
import pyvista as pv
from itertools import chain, repeat, islice
from typing import List, Union


# CXDViz is meant to manage arrays (coords, real,recip) for building structured grids.
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

    def __init__(self, geometry):
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
        self.T = geometry
        self.ssg = pv.StructuredGrid()
        self.arrs = {}
        self.shape = None
        self.voi = None


    def add_array(self, name, array):
        if self.shape is None:
            # the first array added to ssg, initialiize ssg
            self.init_ssg(array.shape)
            self.arrs[name] = array
        elif self.shape == array.shape:
            self.arrs[name] = array
        else:
            raise ValueError(f'Shape mismatch: array has shape {array.shape}, the arrays have shape {self.shape}')


    def get_structured_grid(self, complex_mode="AmpPhase"):
        for arrname, arr in self.arrs.items():
            if np.iscomplexobj(arr):
                match complex_mode:
                    case "AmpPhase":
                        sgname = arrname + "Amp"
                        self.ssg.point_data[sgname] = np.abs(arr).flat
                        sgname = arrname + "Ph"
                        self.ssg.point_data[sgname] = np.angle(arr).flat
                    case "ReIm":
                        sgname = arrname + "Re"
                        self.ssg.point_data[sgname] = arr.real.flat
                        sgname = arrname + "Imag"
                        self.ssg.point_data[sgname] = arr.imag.flat
                    case _:
                        print(f'complex_mode parameter has unsupported value of {complex_mode}')
            else:
                self.ssg.point_data[arrname] = arr.flat
        if self.voi is not None:
            ssg_cropped = self.ssg.extract_subset(self.voi)
        else:
            ssg_cropped = self.ssg

        return ssg_cropped


    def write(self, filename, **kwargs):
        ssg_cropped = self.get_structured_grid(**kwargs)
        ssg_cropped.save(filename)


class Dir_viz(CXDViz):
    def __init__(self, geometry):
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
        super().__init__(geometry[1])       # Trecip, Tdir = geometry


    def init_ssg(self, shape):
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
        q = np.mgrid[0: 1: 1.0 / shape[0], 0: 1: 1.0 / shape[1], 0: 1: 1.0 / shape[2]]
        q.shape = 3, shape[0] * shape[1] * shape[2]

        self.coords = np.dot(self.T, q).transpose()
        self.ssg.points = self.coords
        self.ssg.dimensions = shape[::-1]

        self.shape = shape


class Recip_viz(CXDViz):
    def __init__(self, geometry):
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
        super().__init__(geometry[0])       # Trecip, Tdir = geometry


    def init_ssg(self, shape):
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
        q = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        q.shape = 3, shape[0] * shape[1] * shape[2]

        self.coords = np.dot(self.T, q).transpose()
        self.ssg.points = self.coords
        self.ssg.dimensions = shape[::-1]

        self.shape = shape


def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))


def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)


# lifted from bcdi package by Jerome Carnis
def find_datarange(
        array: np.ndarray,
        plot_margin: Union[int, List[int]] = 10,
        amplitude_threshold: float = 0.1,
        keep_size: bool = False,
) -> List[int]:
    """
    Find the range where data is larger than a threshold.

    It finds the meaningful range of the array where it is larger than the threshold, in
    order to later crop the array to that shape and reduce the memory consumption in
    processing. The range can be larger than the initial data size, which then will need
    to be padded.

    :param array: a non-empty numpy array
    :param plot_margin: user-defined margin to add on each side of the thresholded array
    :param amplitude_threshold: threshold used to define a support from the amplitude
    :param keep_size: set to True in order to keep the dataset full size
    :return: a list of the ranges (centered in the middle of the array) to use in each
     dimension.
    """
    if array.ndim == 0:
        raise ValueError("Empty array provided.")
    if isinstance(plot_margin, int):
        plot_margin = [plot_margin] * array.ndim
    if not isinstance(plot_margin, list):
        raise TypeError(
            f"Expected 'plot_margin' to be a list, got {type(plot_margin)} "
        )
    if len(plot_margin) != array.ndim:
        raise ValueError(
            f"'plot_margin' should be of lenght {array.ndim}, got {len(plot_margin)}"
        )

    if keep_size:
        return [int(val) for val in array.shape]

    support = np.zeros(array.shape)
    support[abs(array) > amplitude_threshold * abs(array).max()] = 1
    non_zero_indices = np.nonzero(support)
    min_half_width_per_axis: List[int] = []
    try:
        for idx, nb_elements in enumerate(array.shape):
            min_half_width_per_axis.append(
                min(
                    min(non_zero_indices[idx]),
                    nb_elements - 1 - max(non_zero_indices[idx]),
                )
            )
    except:
        raise ValueError(f"No voxel above the provided threshold {amplitude_threshold}")
    return [
        int(
            (nb_elements // 2 - min_half_width_per_axis[idx] + plot_margin[idx]) * 2
            + nb_elements % 2
        )
        for idx, nb_elements in enumerate(array.shape)
    ]


def interpolate(sgrid, resolution):
    dims = sgrid.dimensions
    if type(resolution) is list:
        resolution = list(pad(resolution, len(dims), resolution[-1]))
    else:
        resolution = list(pad([resolution], len(dims), [resolution][-1]))
    sg_bounds = sgrid.bounds
    ranges = [sg_bounds[n + 1] - sg_bounds[n] for n in range(0, len(sg_bounds), 2)]
    starts = [sg_bounds[n] for n in range(0, len(sg_bounds), 2)]

    dims = [int(r // resolution[i]) for i, r in enumerate(ranges)]

    image_data = pv.ImageData(
        dimensions=dims,
        origin=starts,
        spacing=resolution
    )

    return image_data.sample(sgrid)


def get_interpolated_arrays(viz, resolution, **kwargs):
    sgrid = viz.get_structured_grid(complex_mode=kwargs['interpolation_mode'])
    return interpolate(sgrid, resolution)


def get_centered_voisize(arrdims, voisize):
    voi = []
    for d in zip(arrdims, voisize):
        voi.append(int(d[0] / 2) - int(d[1] / 2))
        voi.append(int(d[0] / 2) + int(d[1] / 2))
    return voi


def extent_to_slice(extent):
    slice = [np.s_[extent[i]:extent[i + 1]] for i in range(0, len(extent), 2)]
    return tuple(slice)


def get_resolution_deconv(arr, thresh, max_iter=50, deconvdiffbreak=0.1):
    # extract a subregion to make deconvolution faster.
    # when we do PRTF this could also help speed up.  But will need to crop a collection of images.
    voisize = find_datarange(arr, [5, 5, 5])
    voi = get_centered_voisize(arr.shape, voisize)
    slices = extent_to_slice(voi)
    ampscrop = arr[slices]
    contrast = np.where(ampscrop > thresh, 1.0, 0.0)
    kernel = np.full(ampscrop.shape, 0.5)
    # when using cohere_core dvc_utils the package needs to be specified
    dvut.set_lib_from_pkg('np')
    res = dvut.lucy_deconvolution(contrast, ampscrop, kernel, max_iter, diffbreak=deconvdiffbreak)
    # rescaled to more reasonable range.
    res *= 1.0 / res.max()
    # the above not needed as the dvut lucy normalizes by sum
    return res


def make_image_viz(geometry, image, support, config_maps):
    # if it is important to log that some parameters are not configured, do it here
    viz_params = config_maps['config_disp']
    # smooth the image if rampups parameter is greater than 1
    rampups = viz_params.get('rampups', 1)
    if rampups > 1:
        # when using cohere_core dvc_utils the package needs to be specified
        dvut.set_lib_from_pkg('np')
        image = dvut.remove_ramp(image, ups=rampups)

    viz = Dir_viz(geometry)
    # adding the image as a complex array. Can select AmpPhase or ReIm depending on what you want.
    viz.add_array("im", image)

    viz.add_array("support", support)
    unwrap_phase = viz_params.get('unwrap', False)
    if unwrap_phase:
        from skimage import restoration
        unwrapped_phase = restoration.unwrap_phase(np.angle(image))
        viz.add_array("imPhUW", unwrapped_phase)

    if 'imcrop' in viz_params:
        mode = viz_params['imcrop']
        match mode:
            case 'tight':
                arr = np.abs(image)
                imcrop_thresh = viz_params['imcrop_thresh']
                buf = viz_params['imcrop_margin']
                # find_datarange extends an int to the size of arr.dim
                voisize = find_datarange(arr, buf, imcrop_thresh)
                viz.voi = get_centered_voisize(arr.shape[::-1], voisize[::-1])
            case 'fraction':
                dims = image.shape
                arrfrac = list(pad(viz_params['imcrop_fraction'], len(dims), dims[-1]))
                viz.voi = []
                for d in enumerate(zip(dims, arrfrac)):
                    viz.voi.append(int(d[1][0] / 2) - int(d[1][0] * d[1][1] / 2))
                    viz.voi.append(int(d[1][0] / 2) + int(d[1][0] * d[1][1] / 2))
            case _:
                print("No imcrop mode set in config_disp or not supported", mode)

    return viz


def make_recip_viz(geometry, data, ftim):
    viz = Recip_viz(geometry)
    viz.add_array("phasing_data", data)
    viz.add_array('Ftimage', ftim)
    return viz


def make_resolution_viz(geometry, arr, config_maps):
    viz_params = config_maps['config_disp']

    mode = viz_params.get('determine_resolution')
    if mode == 'deconv':
        viz_d = Dir_viz(geometry)
        thresh = viz_params['resolution_deconv_contrast']
        dir_resolution = get_resolution_deconv(arr, thresh)
        viz_d.add_array("resolution", dir_resolution)
        res_ssg = viz_d.get_structured_grid()

        viz_r = Recip_viz(geometry)
        recip_resolution = ut.pad_center(dir_resolution, arr.shape)
        recip_resolution = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(recip_resolution)))
        recip_resolution *= 1 / np.amax(np.absolute(recip_resolution))
        viz_r.add_array("recip_res", recip_resolution)
        return (viz_d, viz_r)
    else:
        print("Resolution mode unknown ", mode)
        return (None, None)


def make_coherence_viz(geometry, coh, dirspace_shape):
    viz_r = Recip_viz(geometry)
    viz_r.add_array("cohrecip", coh)

    viz_d = Dir_viz(geometry)
    coh = ut.pad_center(coh, dirspace_shape)
    coh_d = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(coh)))
    viz_d.add_array("cohdirect", coh_d)

    return (viz_d, viz_r)


##########################################################
# not used
##########################################################
def get_resolution_prtf(viz, config_map):
    pass

def get_image_res(sgrid, deconv_params):
    print(deconv_params)

# I think yudong may have written something that combines this with the data_range.
# need to look at those.  or at least make sure there is dims verification here.
def get_centered_voifraction(arrdims, voifraction):
    voi = []
    print(arrdims, voifraction)
    for d in enumerate(zip(arrdims, voifraction)):
        print(d)
        voi.append(int(d[1][0] / 2) - int(d[1][0] * d[1][1] / 2))
        voi.append(int(d[1][0] / 2) + int(d[1][0] * d[1][1] / 2))
    return voi
