# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import argparse
import cohere_core.controller as rec
import os


def reconstruction(datafile, **kwargs):
    """
    Reconstructs the image of the data in datafile according to arguments the user enters in the script when invoking
    rec.phasing.reconstruction.
    The results:
    image.npy, support.npy, and errors.npy are saved in 'saved_dir' defined in kwargs, or if not defined,
    in the directory of datafile.

    List of supported arguments:
     save_dir : str
        directory where results of reconstruction are saved as npy files. If not present, the reconstruction outcome will be save in the same directory where datafile is.
    processing : str
        the library used when running reconstruction. When the 'auto' option is selected the program will use the best performing library that is available, in the following order: cupy, numpy. The 'cp' option will utilize cupy, and 'np' will utilize numpy. Default is auto.
    device : list of int
        IDs of the target devices. If not defined, it will default to -1 for the OS to select device.
    algorithm_sequence : str
        Mandatory; example: "3* (20*ER + 180*HIO) + 20*ER". Defines algorithm applied in each iteration during modulus projection (ER or HIO) and during modulus (error correction or partial coherence correction). The "*" character means repeat, and the "+" means add to the sequence. The sequence may contain single brackets defining a group that will be repeated by the preceding multiplier. The alphabetic entries: 'ER', 'ERpc', 'HIO', 'HIOpc' define algorithms used in iterations.
        If the defined algorithm contains 'pc' then during modulus operation a partial coherence correction is applied,  but only if partial coherence (pc) feature is activated. If not activated, the phasing will use error correction instead.
    hio_beta : float
         multiplier used in hio algorithm
    twin_trigger : list
         example: [2]. Defines at which iteration to cut half of the array (i.e. multiply by 0s)
    twin_halves : list
        defines which half of the array is zeroed out in x and y dimensions. If 0, the first half in that dimension is zeroed out, otherwise, the second half.
    shrink_wrap_trigger : list
        example: [1, 1]. Defines when to update support array using the parameters below.
    shrink_wrap_type : str
        supporting "GAUSS" only. Defines which algorithm to use for shrink wrap.
    shrink_wrap_threshold : float
        only point with relative intensity greater than the threshold are selected
    shrink_wrap_gauss_sigma : float
        used to calculate the Gaussian filter
    initial_support_area : list
        If the values are fractional, the support area will be calculated by multiplying by the data array dimensions. The support will be set to 1s to this dimensions centered.
    phc_trigger : list
        defines when to update support array using the parameters below by applaying phase constrain.
    phc_phase_min : float
        point with phase below this value will be removed from support area
    phc_phase_max : float
        point with phase over this value will be removed from support area
    pc_interval : int
        defines iteration interval to update coherence.
    pc_type : str
        partial coherence algorithm. 'LUCY' type is supported.
    pc_LUCY_iterations : int
        number of iterations used in Lucy algorithm
    pc_LUCY_kernel : list
        coherence kernel area.
    lowpass_filter_trigger : list
        defines when to apply lowpass filter using the parameters below.
    lowpass_filter_sw_threshold : float
        used in Gass type shrink wrap when applying lowpass filter.
    lowpass_filter_range : list
        used when applying low resolution data filter while iterating. The values are linespaced for lowpass filter iterations from first value to last. The filter is gauss with sigma of linespaced value. If only one number given, the last value will default to 1.
    average_trigger : list
        defines when to apply averaging. Negative start means it is offset from the last iteration.
    progress_trigger : list of int
        defines when to print info on the console. The info includes current iteration and error.

    :param datafile: data file
    :param kwargs: may contain debug. The no_verify is implied.
    :return:
    """
    datafile = datafile.replace(os.sep, '/')
    if not os.path.isfile(datafile):
        print(f'no file found {datafile}')
        return

    # create dictionary with the parameters applied for reconstruction
    params = {
        'algorithm_sequence' : '3*(20*ER+180*HIO)+20*ER',
        'shrink_wrap_trigger' : [1, 1],
        'shrink_wrap_type' : "GAUSS",
        'shrink_wrap_threshold' : 0.1,
        'shrink_wrap_gauss_sigma' : 1.0,
        'twin_trigger' : [2],
        'progress_trigger' : [0, 20],
        'save_dir' : "results"
        }

    # select the package to run reconstruction: 'cp' for cupy, 'np' for numpy, and 'torch' for torch.
    pkg = 'cp'
    # Select the GPU id to use
    device = 0

    worker = rec.create_rec(params, datafile, pkg, device, **kwargs)
    if worker is None:
        return

    if worker.iterate() < 0:
        return

    if 'save_dir' in kwargs:
        save_dir = kwargs['save_dir']
    else:
        save_dir, filename = os.path.split(datafile)

    worker.save_res(save_dir)


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("datafile", help="data file name. It should be either tif file or numpy.")
        parser.add_argument("--debug", action="store_true",
                            help="if True the exceptions are not handled")
        args = parser.parse_args()
        reconstruction(args.datafile, debug=args.debug)


if __name__ == "__main__":
    exit(main())
