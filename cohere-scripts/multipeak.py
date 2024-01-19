# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This user script processes the multi-peak reconstructed image for visualization.
After the script is executed the experiment directory will contain image.vti file containing density, support, and the
three components of atomic displacement.
"""

__author__ = "Nick Porter"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['rotate_peaks',
           'twin_matrix',
           'center_mp',
           'write_vti',
           'process_dir']

from pathlib import Path
import numpy as np
from tvtk.api import tvtk
import scipy.ndimage as ndi
from multiprocessing import Process
from prep_helper import Preparer, combine_scans, write_prep_arr
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.spatial.transform import Rotation as R
import cohere_core.utilities as ut
from beamlines.aps_34idc import instrument as instr, diffractometers as diff, detectors as det


def rotate_peaks(prep_obj, data, scans, o_twin):
    print("rotating diffraction pattern")
    config_map = ut.read_config(ut.join(prep_obj.experiment_dir, 'conf', 'config_instr'))
    config_map['multipeak'] = True
    instr_obj = instr.Instrument()
    instr_obj.initialize(config_map, scans[-1])

    shape = data.shape
    B_recip, _ = instr_obj.get_geometry(shape, xtal=True)
    B_recip = np.stack([B_recip[1, :], B_recip[0, :], B_recip[2, :]])
    voxel_size = np.abs(np.linalg.det(B_recip))**(1/3)

    B_recip = o_twin @ B_recip

    x = np.arange(-shape[0] // 2, shape[0] // 2)  # define old grid
    y = np.arange(-shape[1] // 2, shape[1] // 2)
    z = np.arange(-shape[2] // 2, shape[2] // 2)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    old_grid = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])  # flatten grid into a 3xN array

    new_points = B_recip @ old_grid
    min_max = np.array([[np.min(row), np.max(row)] for row in new_points])

    a = np.arange(min_max[0, 0], min_max[0, 1], voxel_size)  # define new grid from the smallest to largest value
    b = np.arange(min_max[1, 0], min_max[1, 1], voxel_size)  # ensure that the step size is that of the unit cell
    c = np.arange(min_max[2, 0], min_max[2, 1], voxel_size)  # edge length
    aa, bb, cc = np.meshgrid(a, b, c, indexing='ij')

    new_grid = np.vstack([aa.ravel(), bb.ravel(), cc.ravel()])  # flatten new grid
    interp_points = np.linalg.inv(B_recip) @ new_grid  # transform new grid points to old grid

    interp = RGI((x, y, z), data, fill_value=0, bounds_error=False)  # feed interpolator node values from data
    data = interp(interp_points.T).reshape((a.shape[0], b.shape[0], c.shape[0]))  # interpolate values on new grid

    final_size = prep_obj.final_size
    shp = np.array([final_size, final_size, final_size]) // 2
    shp1 = np.array(data.shape) // 2
    pad = shp - shp1
    pad[pad < 0] = 0

    data = np.pad(data, [(pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2])])
    shp1 = np.array(data.shape) // 2
    start, end = shp1 - shp, shp1 + shp
    return data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]


def twin_matrix(hklin, hklout, twin_plane, sample_axis):
    r1 = ut.normalize(hklin)
    r3 = ut.normalize(np.cross(hklin, hklout))
    r2 = ut.normalize(np.cross(r3, r1))
    rmat = np.stack([r1, r2, r3])

    twin_plane = rmat @ twin_plane
    theta = np.arccos(twin_plane @ sample_axis / (np.linalg.norm(twin_plane)*np.linalg.norm(sample_axis)))
    vec = ut.normalize(np.cross(twin_plane, sample_axis))

    return R.from_rotvec(vec * -theta).as_matrix()


class MultPeakPreparer(Preparer):
    def __init__(self, prep_obj):
        """
        Creates PrepData instance for beamline aps_34idc. Sets fields to configuration parameters.
        Parameters
        ----------
        experiment_dir : str
            directory where the files for the experiment processing are created
        Returns
        -------
        PrepData object
        """
        super().__init__(prep_obj)
        try:
            self.o_twin = twin_matrix(prep_obj.hkl_in, prep_obj.hkl_out, prep_obj.twin_plane,
                                 prep_obj.sample_axis)
        except KeyError:
            self.o_twin = np.identity(3)


    def get_batches(self):
        batches = super().get_batches()
        for batch in batches:
            # figure order of the batches relative to params stored in prep_obj
            index = batch[1][0]  # first index of scan in batch
            i = 0
            while index > self.prep_obj.scan_ranges[i][-1]:
                i += 1
            batch.append(i)
        return batches


    def prepare(self, batches):
        processes = []
        for i in range(len(batches)):
            dirs = batches[i][0]
            scans = batches[i][1]
            order = batches[i][2]
            conf_scans = f'{str(self.prep_obj.scan_ranges[order][0])}-{str(self.prep_obj.scan_ranges[order][1])}'
            orientation = self.prep_obj.orientations[order]
            orientation = str(orientation[0]) + str(orientation[1]) + str(orientation[2])
            save_dir = ut.join(self.prep_obj.experiment_dir, f'mp_{conf_scans}_{orientation}', 'preprocessed_data')
            p = Process(target=self.process_batch,
                        args=(dirs, scans, save_dir, 'prep_data.tif'))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


    def process_batch(self, dirs, scans, save_dir, filename):
        batch_arr = combine_scans(self.prep_obj, dirs, scans)
        batch_arr = self.prep_obj.det_obj.clear_seam(batch_arr)
        data = rotate_peaks(self.prep_obj, batch_arr, scans, self.o_twin)
        write_prep_arr(data, save_dir, filename)


def center_mp(image, support):
    """
    Shifts the image and support arrays so the center of mass is in the center of array.
    Parameters
    ----------
    image, support : ndarray, ndarray
        image and support arrays to evaluate and shift
    Returns
    -------
    image, support : ndarray, ndarray
        shifted arrays
    """
    density = image[0]
    shape = density.shape
    max_coordinates = list(np.unravel_index(np.argmax(density), shape))
    for i in range(len(max_coordinates)):
        image[0] = np.roll(image[0], int(shape[i] / 2) - max_coordinates[i], i)
        image[1] = np.roll(image[1], int(shape[i] / 2) - max_coordinates[i], i)
        image[2] = np.roll(image[2], int(shape[i] / 2) - max_coordinates[i], i)
        image[3] = np.roll(image[3], int(shape[i] / 2) - max_coordinates[i], i)
        support = np.roll(support, int(shape[i] / 2) - max_coordinates[i], i)

    com = ndi.center_of_mass(density * support)
    # place center of mass in the center
    for i in range(len(shape)):
        image[0] = np.roll(image[0], int(shape[i] / 2 - com[i]), axis=i)
        image[1] = np.roll(image[1], int(shape[i] / 2 - com[i]), axis=i)
        image[2] = np.roll(image[2], int(shape[i] / 2 - com[i]), axis=i)
        image[3] = np.roll(image[3], int(shape[i] / 2 - com[i]), axis=i)
        support = np.roll(support, int(shape[i] / 2 - com[i]), axis=i)

    # set center displacement to zero, use as a reference
    half = np.array(shape) // 2
    for i in [1, 2, 3]:
        image[i] = image[i] - image[i, half[0], half[1], half[2]]

    return image, support


def write_vti(data, px, savedir):
    # TODO get the voxel size directly from config_mp
    # Create the vtk object for the data
    print("Preparing VTK data")
    grid = tvtk.ImageData(dimensions=data[0].shape, spacing=(px, px, px))
    # Set the data to the image/support/distortion
    for img, name in zip(data, ["density", "u_x", "u_y", "u_z", "support"]):
        arr = tvtk.DoubleArray()
        arr.from_array(img.ravel())
        arr.name = name
        grid.point_data.add_array(arr)

    # print("Saving VTK")
    # Create the writer object
    writer = tvtk.XMLImageDataWriter(file_name=f"{savedir}/full_data.vti")
    writer.set_input_data(grid)
    # Save the data
    writer.write()
    print(f"saved file: {savedir}/full_data.vti")


def process_dir(res_dir, rampups=1, make_twin=False):
    """
    Loads arrays from files in results directory. If reciprocal array exists, it will save reciprocal info in tif
    format.

    Parameters
    ----------
    res_dir : str
        the directory where phasing results are saved
    rampups : int
        factor to apply to rampups operation, i.e. smoothing the image
    make_twin : bool
        if True visualize twin
    """
    save_dir = Path(res_dir.replace('_phasing', '_viz'))
    res_dir = Path(res_dir)
    # create dir if does not exist
    print(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()
    for f in save_dir.iterdir():
        f.unlink()

    image = np.load(f"{res_dir}/image.npy")
    image = np.moveaxis(image, 3, 0)
    image[0] = image[0] / np.max(image[0])
    support = np.load(f"{res_dir}/support.npy")

    image, support = center_mp(image, support)
    if rampups > 1:
        image = ut.remove_ramp(image, ups=rampups)

    write_vti(np.concatenate((image, support[None])), 1.0, save_dir)

    if make_twin:
        image = np.flip(image)
        if support is not None:
            support = np.flip(support)
            image, support = center_mp(image, support)
        if rampups > 1:
            image = ut.remove_ramp(image, ups=rampups)
        write_vti(np.concatenate((image, support[None])), 1.0, save_dir)
