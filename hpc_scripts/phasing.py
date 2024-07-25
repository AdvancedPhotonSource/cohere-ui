# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################


"""
cohere_core.phasing
===================

Provides phasing capabilities for the Bragg CDI data.
The software can run code utilizing different library, such as numpy and cupy. User configures the choice depending on hardware and installed software.

"""

from pathlib import Path
import time
import os
import argparse
import cohere_core.utilities.dvc_utils as dvut
import cohere_core.utilities.utils as ut
import cohere_core.controller.op_flow as of
import cohere_core.controller.features as ft
from mpi4py import MPI


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['set_lib_from_pkg',
           'Rec']


def set_lib_from_pkg(pkg):
    global devlib

    # get the lib object
    devlib = ut.set_lib(pkg)
    # pass the lib object to the features associated with this reconstruction
    ft.set_lib(devlib)
    # the utilities are not associated with reconstruction and the initialization of lib is independent
    dvut.set_lib_from_pkg(pkg)


class Rec:
    """
    cohere_core.phasing.reconstruction(self, params, data_file)

    Class, performs phasing using iterative algorithm.

    params : dict
        parameters used in reconstruction. Refer to x for parameters description
    data_file : str
        name of file containing data to be reconstructed

    """
    __all__ = []
    def __init__(self, params, data_file, pkg):
        set_lib_from_pkg(pkg)
        self.iter_functions = [self.next,
                               self.lowpass_filter_operation,
                               self.reset_resolution,
                               self.shrink_wrap_operation,
                               self.phc_operation,
                               self.to_reciprocal_space,
                               self.new_func_operation,
                               self.pc_operation,
                               self.pc_modulus,
                               self.modulus,
                               self.set_prev_pc,
                               self.to_direct_space,
                               self.er,
                               self.hio,
                               self.new_alg,
                               self.twin_operation,
                               self.average_operation,
                               self.progress_operation]

        params['init_guess'] = params.get('init_guess', 'random')
        if params['init_guess'] == 'AI_guess':
            if 'AI_threshold' not in params:
                params['AI_threshold'] = params['shrink_wrap_threshold']
            if 'AI_sigma' not in params:
                params['AI_sigma'] = params['shrink_wrap_gauss_sigma']
        params['reconstructions'] = params.get('reconstructions', 1)
        params['hio_beta'] = params.get('hio_beta', 0.9)
        params['initial_support_area'] = params.get('initial_support_area', (.5, .5, .5))
        if 'twin_trigger' in params:
            params['twin_halves'] = params.get('twin_halves', (0, 0))
        if 'pc_interval' in params and 'pc' in params['algorithm_sequence']:
            self.is_pcdi = True
        else:
            self.is_pcdi = False
        # finished setting defaults
        self.params = params
        self.data_file = data_file
        self.ds_image = None
        self.need_save_data = False
        self.saved_data = None
        self.er_iter = False  # Indicates whether the last iteration done was ER, used in CoupledRec

    def init_dev(self, device_id=-1):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if device_id != -1:
            self.dev = device_id
            if device_id != -1:
                try:
                    devlib.set_device(device_id)
                except Exception as e:
                    print(e)
                    print('may need to restart GUI')
                    return -1

        if self.data_file.endswith('tif') or self.data_file.endswith('tiff'):
            try:
                data_np = ut.read_tif(self.data_file)
                data = devlib.from_numpy(data_np)
            except Exception as e:
                print(e)
                return -1
        elif self.data_file.endswith('npy'):
            try:
                data = devlib.load(self.data_file)
            except Exception as e:
                print(e)
                return -1
        else:
            print('no data file found')
            return -1
        # in the formatted data the max is in the center, we want it in the corner, so do fft shift
        self.data = devlib.fftshift(devlib.absolute(data))
        self.dims = devlib.dims(self.data)
        print('data shape', self.dims)

        if self.need_save_data:
            self.saved_data = devlib.copy(self.data)
            self.need_save_data = False

        return 0


    def init(self, dir=None, alpha_dir=None, gen=None):
        def create_feat_objects(params, trig_op_info):
            if 'shrink_wrap_trigger' in params:
                self.shrink_wrap_obj = ft.create('shrink_wrap', params, trig_op_info)
                if self.shrink_wrap_obj is None:
                    print('failed to create shrink wrap object')
                    return False
            if 'phc_trigger' in params:
                self.phc_obj = ft.create('phc', params, trig_op_info)
                if self.phc_obj is None:
                    print('failed to create phase constrain object')
                    return False
            if 'lowpass_filter_trigger' in params:
                self.lowpass_filter_obj = ft.create('lowpass_filter', params, trig_op_info)
                if self.lowpass_filter_obj is None:
                    print('failed to create lowpass filter object')
                    return False
            return True

        if self.ds_image is not None:
            first_run = False
        elif dir is None or not os.path.isfile(ut.join(dir, 'image.npy')):
            self.ds_image = devlib.random(self.dims, dtype=self.data.dtype)
            first_run = True
        else:
            self.ds_image = devlib.load(ut.join(dir, 'image.npy'))
            first_run = False

        # When running GA the lowpass filter, phc, and twin triggers should be active only during first
        # generation. The code below inactivates the triggers in subsequent generations.
        # This will be removed in the future when each generation will have own configuration.
        if not first_run:
            self.params.pop('lowpass_filter_trigger', None)
            self.params.pop('phc_trigger', None)
            self.params.pop('twin_trigger', None)

        self.flow_items_list = [f.__name__ for f in self.iter_functions]

        self.is_pc, flow, feats = of.get_flow_arr(self.params, self.flow_items_list, gen)
        if flow is None:
            return -1

        self.flow = []
        (op_no, self.iter_no) = flow.shape
        for i in range(self.iter_no):
            for j in range(op_no):
                if flow[j, i] > 0:
                    self.flow.append(self.iter_functions[j])

        self.aver = None
        self.iter = -1
        self.errs = []
        self.gen = gen
        self.prev_dir = dir
        self.alpha_dir = alpha_dir

        # create or get initial support
        if dir is None or not os.path.isfile(ut.join(dir, 'support.npy')):
            init_support = [int(self.params['initial_support_area'][i] * self.dims[i]) for i in range(len(self.dims))]
            center = devlib.full(init_support, 1)
            self.support = dvut.pad_around(center, self.dims, 0)
        else:
            self.support = devlib.load(ut.join(dir, 'support.npy'))

        if self.is_pc:
            self.pc_obj = ft.Pcdi(self.params, self.data, dir)
        # create the object even if the feature inactive, it will be empty
        # If successful, it will return True, otherwise False
        if not create_feat_objects(self.params, feats):
            return -1

        # for the fast GA the data needs to be saved, as it would be changed by each lr generation
        # for non-fast GA the Rec object is created in each generation with the initial data
        if self.saved_data is not None:
            if self.params['low_resolution_generations'] > self.gen:
                self.data = devlib.gaussian_filter(self.saved_data, self.params['ga_lpf_sigmas'][self.gen])
            else:
                self.data = self.saved_data
        else:
            if self.gen is not None and self.params['low_resolution_generations'] > self.gen:
                self.data = devlib.gaussian_filter(self.data, self.params['ga_lpf_sigmas'][self.gen])

        if 'lowpass_filter_range' not in self.params or not first_run:
            self.iter_data = self.data
        else:
            self.iter_data = devlib.copy(self.data)

        if (first_run):
            max_data = devlib.amax(self.data)
            self.ds_image *= dvut.get_norm(self.ds_image) * max_data

            # the line below are for testing to set the initial guess to support
            # self.ds_image = devlib.full(self.dims, 1.0) + 1j * devlib.full(self.dims, 1.0)

            self.ds_image *= self.support
        return 0


    def iterate(self):
        self.iter = -1
        start_t = time.time()
        try:
            for f in self.flow:
                f()
        except Exception as error:
            print(error)
            return -1

        if devlib.hasnan(self.ds_image):
            print('reconstruction resulted in NaN')
            return -1

        print('iterate took ', (time.time() - start_t), ' sec')

        if self.aver is not None:
            ratio = self.get_ratio(devlib.from_numpy(self.aver), devlib.absolute(self.ds_image))
            self.ds_image *= ratio / self.aver_iter

        mx = devlib.amax(devlib.absolute(self.ds_image))
        self.ds_image = self.ds_image / mx

        return 0

    def save_res(self, save_dir, only_image=False):
        # center image's center_of_mass and sync support
        self.ds_image, self.support = dvut.center_sync(self.ds_image, self.support)

        from array import array

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        devlib.save(ut.join(save_dir, 'image'), self.ds_image)
        ut.save_tif(devlib.to_numpy(self.ds_image), ut.join(save_dir, 'image.tif'))

        if only_image:
            return 0

        devlib.save(ut.join(save_dir, 'support'), self.support)
        if self.is_pc:
            devlib.save(ut.join(save_dir, 'coherence'), self.pc_obj.kernel)
        errs = array('f', self.errs)

        with open(ut.join(save_dir, 'errors.txt'), 'w+') as err_f:
            err_f.write('\n'.join(map(str, errs)))

        devlib.save(ut.join(save_dir, 'errors'), errs)

        metric = dvut.all_metrics(self.ds_image, self.errs)
        with open(ut.join(save_dir, 'metrics.txt'), 'w+') as f:
            f.write(str(metric))

        return 0

    def get_metric(self, metric_type):
        return dvut.all_metrics(self.ds_image, self.errs)

    def next(self):
        self.iter = self.iter + 1

    def lowpass_filter_operation(self):
        args = (self.data, self.iter, self.ds_image)
        (self.iter_data, self.support) = self.lowpass_filter_obj.apply_trigger(*args)

    def reset_resolution(self):
        self.iter_data = self.data

    def shrink_wrap_operation(self):
        args = (self.ds_image,)
        self.support = self.shrink_wrap_obj.apply_trigger(*args)

    def phc_operation(self):
        args = (self.ds_image,)
        self.support *= self.phc_obj.apply_trigger(*args)

    def to_reciprocal_space(self):
        self.rs_amplitudes = devlib.ifft(self.ds_image)

    def new_func_operation(self):
        self.params['new_param'] = 1
        print(f'in new_func_trigger, new_param {self.params["new_param"]}')

    def pc_operation(self):
        self.pc_obj.update_partial_coherence(devlib.absolute(self.rs_amplitudes))

    def pc_modulus(self):
        abs_amplitudes = devlib.absolute(self.rs_amplitudes)
        converged = self.pc_obj.apply_partial_coherence(abs_amplitudes)
        ratio = self.get_ratio(self.iter_data, devlib.absolute(converged))
        error = dvut.get_norm(
            devlib.where(devlib.absolute(converged) != 0.0, devlib.absolute(converged) - self.iter_data, 0.0)) / dvut.get_norm(self.iter_data)
        self.errs.append(error)
        self.rs_amplitudes *= ratio

    def modulus(self):
        ratio = self.get_ratio(self.iter_data, devlib.absolute(self.rs_amplitudes))
        error = dvut.get_norm(devlib.where((self.rs_amplitudes != 0), (devlib.absolute(self.rs_amplitudes) - self.iter_data),
                                      0)) / dvut.get_norm(self.iter_data)
        self.errs.append(error)
        self.rs_amplitudes *= ratio

    def set_prev_pc(self):
        self.pc_obj.set_previous(devlib.absolute(self.rs_amplitudes))

    def to_direct_space(self):
        self.ds_image_raw = devlib.fft(self.rs_amplitudes)

    def er(self):
        self.er_iter = True
        self.ds_image = self.ds_image_raw * self.support

    def hio(self):
        self.er_iter = False
        combined_image = self.ds_image - self.ds_image_raw * self.params['hio_beta']
        self.ds_image = devlib.where((self.support > 0), self.ds_image_raw, combined_image)

    def new_alg(self):
        self.ds_image = 2.0 * (self.ds_image_raw * self.support) - self.ds_image_raw

    def twin_operation(self):
        # TODO this will work only for 3D array, but will the twin be used for 1D or 2D?
        # com = devlib.center_of_mass(devlib.absolute(self.ds_image))
        # sft = [int(self.dims[i] / 2 - com[i]) for i in range(len(self.dims))]
        # self.ds_image = devlib.shift(self.ds_image, sft)
        dims = devlib.dims(self.ds_image)
        half_x = int((dims[0] + 1) / 2)
        half_y = int((dims[1] + 1) / 2)
        if self.params['twin_halves'][0] == 0:
            self.ds_image[half_x:, :, :] = 0
        else:
            self.ds_image[: half_x, :, :] = 0
        if self.params['twin_halves'][1] == 0:
            self.ds_image[:, half_y:, :] = 0
        else:
            self.ds_image[:, : half_y, :] = 0

    def average_operation(self):
        if self.aver is None:
            self.aver = devlib.to_numpy(devlib.absolute(self.ds_image))
            self.aver_iter = 1
        else:
            self.aver = self.aver + devlib.to_numpy(devlib.absolute(self.ds_image))
            self.aver_iter += 1

    def progress_operation(self):
        print(f'------iter {self.iter}   error {self.errs[-1]}')

    def get_ratio(self, dividend, divisor):
        ratio = devlib.where((divisor > 1e-9), dividend / divisor, 0.0)
        return ratio


class TeRec(Rec):
    """
    Coherent diffractive imaging of time-evolving samples with improved temporal resolution

    params : dict
        parameters used in reconstruction. Refer to x for parameters description
    data_file : str
        name of file containing data

    """

    def __init__(self, params, data_file, comm, pkg='cp'):
        super().__init__(params, data_file, pkg)

        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.weight = .1

        print('data file, rank', data_file, self.rank)


    def er(self):
        self.comm.Barrier()
        if self.rank != 0:
            self.comm.send(self.ds_image, dest=self.rank-1)
        if self.rank != self.size -1:
            ds_image_next = self.comm.recv(source=self.rank+1)
        self.comm.Barrier()
        if self.rank != self.size -1:
            self.comm.send(self.ds_image, dest=self.rank+1)
        if self.rank != 0:
            ds_image_prev = self.comm.recv(source=self.rank-1)

        if self.rank == 0 or self.rank == self.size - 1:
            self.ds_image = self.ds_image_raw * self.support
        else:
            self.ds_image = (1/(1+2*self.weight)) * self.support * (self.ds_image_raw +
            self.weight * (ds_image_prev + ds_image_next))


    def hio(self):
        self.comm.Barrier()
        if self.rank > 1:
            self.comm.send(self.ds_image, dest=self.rank-1)
        if self.rank > 0 and self.rank != self.size - 1:
            ds_image_next = self.comm.recv(source=self.rank+1)
        self.comm.Barrier()
        if self.rank < self.size - 2:
            self.comm.send(self.ds_image, dest=self.rank+1)
        if self.rank < self.size - 1 and self.rank != 0:
            ds_image_prev = self.comm.recv(source=self.rank-1)

        if self.rank == 0 or self.rank == self.size - 1:
            combined_image = self.ds_image - self.ds_image_raw * self.params['hio_beta']
            self.ds_image = devlib.where((self.support > 0), self.ds_image_raw, combined_image)
        else:
            combined_image = self.ds_image - self.ds_image_raw * self.params['hio_beta']
            corr = self.weight * self.support * (2 * (self.ds_image) - (ds_image_prev + ds_image_next))
            self.ds_image = devlib.where((self.support > 0), self.ds_image_raw, combined_image) - corr


def time_evolving_rec():
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument("conf", help="conf")
    parser.add_argument("datafile_dir", help="directory with datafiles")
    args = parser.parse_args()

    params = ut.read_config(args.conf)
    params['weight'] = 0.1

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    data_files = ast.literal_eval(args.datafile_dir)
    datafile = data_files[rank]
    worker = TeRec(params, datafile, comm)

#    worker.comm = comm
#    worker.size = size
#    worker.rank = rank

    worker.init_dev()
    ret_code = worker.init()
    if ret_code < 0:
        print ('reconstruction failed, check algorithm sequence and triggers in configuration')
        return

    ret_code = worker.iterate()
    if ret_code < 0:
        print ('reconstruction failed during iterations')
        return

    if 'save_dir' in params:
        save_dir = params['save_dir']
    else:
        save_dir, filename = os.path.split(datafile)
        save_dir = save_dir.replace('phasing_data', 'results_phasing')
    print('save_dir', save_dir)
    worker.save_res(save_dir)

if __name__ == "__main__":
    exit(time_evolving_rec())
