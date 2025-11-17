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
import math


# CXDViz is meant to manage arrays (coords, real,recip) for building structured grids.
class CXDViz:
    """
    CXDViz(self, geometry)
    ===================================
    Class, generates files for visualization from reconstructed suite.

    geometry : tuple of arrays
        arrays containing geometry in reciprocal and direct space
    """

    def __init__(self, geometry):
        """
        The constructor creates objects assisting with visualization.
        Parameters
        ----------
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
            # the first array added to ssg, initialize ssg
            self.init_ssg(array.shape)
            self.arrs[name] = array
        elif self.shape == array.shape:
            self.arrs[name] = array
            # Allow it to be ok if the array is 1D but has the correct number of elements.  This is mostly for strain calc.
        elif array.shape[0] == math.prod(self.shape):
            self.arrs[name] = array
        else:
            print('prod', math.prod(self.shape), array.shape[0], math.prod(array.shape))
            raise ValueError(f'Shape mismatch: array has shape {array.shape}, the arrays have shape {self.shape}')

    def get_structured_grid(self, complex_mode="AmpPhase", arrstoinclude=None):
        # Allow to ask for a structured grid with only some arrays (pass list of names).
        if arrstoinclude is None:
            arrs = self.arrs.items()
        else:
            # if arrs is provided, use it instead of self.arrs
            for arrname in arrstoinclude:
                if arrname not in self.arrs:
                    raise ValueError(f'Array {arrname} not found in the viz arrays')
            arrs = [(arrname, self.arrs[arrname]) for arrname in arrstoinclude]

        for arrname, arr in arrs:
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

    def get_COM(self, array_name):
        """
        Get the center of mass of the array with the given name.
        Parameters
        ----------
        array_name : str
            name of the array to compute COM for
        Returns
        -------
        tuple
            center of mass coordinates
        """
        if array_name not in self.arrs:
            raise ValueError(f'Array {array_name} not found')
        arr = self.arrs[array_name]
        # taking the abs of the array is assuming something.
        masses = np.abs(arr).flatten().reshape(-1, 1)

        com = (self.coords * masses).sum(axis=0) / masses.sum()
        return com


class Dir_viz(CXDViz):
    def __init__(self, geometry):
        """
        The constructor creates objects assisting with visualization.
        Parameters
        ----------
        geometry : tuple of arrays
            arrays containing geometry in reciprocal and direct space
        Returns
        -------
        constructed object
        """
        super().__init__(geometry[1])  # Trecip, Tdir = geometry

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
        geometry : tuple of arrays
            arrays containing geometry in reciprocal and direct space
        Returns
        -------
        constructed object
        """
        super().__init__(geometry[0])  # Trecip, Tdir = geometry

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


def make_image_viz(geometry, image, support, config_maps, ds):
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

    if support is not None:
        viz.add_array("support", support)

    unwrap_phase = viz_params.get('unwrap', False)
    if unwrap_phase:
        from skimage import restoration
        unwrapped_phase = restoration.unwrap_phase(np.angle(image))
        viz.add_array("imPhUW", unwrapped_phase)
    else:
        unwrapped_phase = None

    # in principle the displacement should be computed from the unwrapped phase.
    # but one could also use the raw phase.  If there are no wraps it would be fine.
    # maybe we need to provide a warning if the phase is not unwrapped.
    # also, need the displacment field if strain is asked for, so switch on that param.
    if viz_params.get('Bragg_displacement', None) is not None:
        # convert phase to displacement
        if viz_params['Bragg_displacement'] == 'Q':
            myq = geometry[2]
            d_spacing_on2pi = 1.0 / np.linalg.norm(myq)
            ds['displacement (from Q)'] = d_spacing_on2pi * 2 * np.pi
        if unwrapped_phase is not None:
            displacementfield = unwrapped_phase * d_spacing_on2pi / 10.0  # convert to nanometers since coords are nm and derivative needs both to have same units
        else:
            displacementfield = np.angle(image) * d_spacing_on2pi / 10.0  # convert to nanometers

        viz.add_array("Displacement", displacementfield)

    if viz_params.get("compute_strain", False):
        if 'Displacement' not in viz.arrs.keys():
            raise ValueError(
                f'Unable processing of strain with displacement not set. Activate the displacement in GUI or when running from command line, set Bragg_displacement parameter, exiting')

        strainsg = viz.get_structured_grid(arrstoinclude=['Displacement']).compute_derivative(scalars='Displacement',
                                                                                              gradient='DisplacementGradient')
        myq = geometry[2]
        disp_grad = np.dot(strainsg.point_data['DisplacementGradient'] / 10.0,
                           myq)  # divide by 10 to convert from Angstrom to nm
        viz.add_array('Qstrain', disp_grad)

    if 'crop_type' in viz_params:
        mode = viz_params['crop_type']
        match mode:
            case 'tight':
                arr = np.abs(image)
                crop_thresh = viz_params['crop_thresh']
                buf = viz_params['crop_margin']
                # find_datarange extends an int to the size of arr.dim
                voisize = find_datarange(arr, buf, crop_thresh)
                viz.voi = get_centered_voisize(arr.shape[::-1], voisize[::-1])
            case 'fraction':
                dims = image.shape
                arrfrac = list(pad(viz_params['crop_fraction'], len(dims), dims[-1]))
                viz.voi = []
                for d in enumerate(zip(dims, arrfrac)):
                    viz.voi.append(int(d[1][0] / 2) - int(d[1][0] * d[1][1] / 2))
                    viz.voi.append(int(d[1][0] / 2) + int(d[1][0] * d[1][1] / 2))
            case _:
                print("No crop_type mode set in config_disp or not supported", mode)

    return viz


def make_geometry_vectors_viz(geometry, position=[0, 0, 0]):
    Q = geometry[2]
    ki = geometry[3]
    kf = geometry[4]
    usg = pv.UnstructuredGrid()
    usg.points = [position, position, position]
    usg.point_data['vectors'] = [Q, ki, kf]
    return usg


def make_recip_viz(geometry, data, ftim):
    viz = Recip_viz(geometry)
    viz.add_array("phasing_data", data)
    viz.add_array('Ftimage', ftim)
    return viz


def make_resolution_viz(geometry, arr, config_maps):
    viz_params = config_maps['config_disp']

    mode = viz_params.get('determine_resolution_type')
    if mode == 'deconv':
        viz_d = Dir_viz(geometry)
        thresh = viz_params['resolution_deconv_contrast']
        dir_resolution = get_resolution_deconv(arr, thresh)
        viz_d.add_array("resolution", dir_resolution)

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
