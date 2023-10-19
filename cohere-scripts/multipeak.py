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
__all__ = ['calc_geometry',
           'rotate_peaks',
           'twin_matrix',
           'center_mp',
           'write_vti',
           'process_dir']

from pathlib import Path
import numpy as np
from tvtk.api import tvtk
from multiprocessing import Process
from prep_helper import Preparer, combine_scans, write_prep_arr
from matplotlib import pyplot as plt
from skimage import transform
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation as R
import cohere_core.utilities as ut
from beamlines.aps_34idc import instrument as instr, diffractometers as diff, detectors as det


def calc_geometry(prep_obj, scans, shape, o_twin):
    """Calculates the rotation matrix and voxel size for a given peak"""
    config_map = ut.read_config(prep_obj.experiment_dir + '/conf/config_instr')
    config_map['multipeak'] = True
    instr_obj = instr.Instrument()
    instr_obj.initialize(config_map, scans[-1])
    B_recip, _ = instr_obj.get_geometry(shape, xtal=True)
    B_recip = np.stack([B_recip[1, :], B_recip[0, :], B_recip[2, :]])
    rs_voxel_size = np.max([np.linalg.norm(B_recip[:, i]) for i in range(3)])  # Units are inverse nanometers
    B_recip = o_twin @ B_recip
    return B_recip, rs_voxel_size


def rolloff3d(shape, sigma):
    mask = np.zeros(shape)
    mask[3*sigma:-3*sigma, 3*sigma:-3*sigma, 3*sigma:-3*sigma] = 1
    submask = np.zeros((2*sigma+1, 2*sigma+1, 2*sigma+1))
    a, b, c = np.mgrid[-sigma:sigma+1, -sigma:sigma+1, -sigma:sigma+1]
    submask[a**2+b**2+c**2 < sigma**2] = 1

    mask = ndi.binary_dilation(mask, submask).astype(float)
    mask = ndi.gaussian_filter(mask, sigma)
    return mask


def pad_to_cube(arr):
    padx, pady, padz = (np.max(arr.shape) - np.array(arr.shape)) // 2
    arr = np.pad(arr, ((padx, padx), (pady, pady), (padz, padz)))
    if len(np.unique(arr.shape)) != 1:
        padx, pady, padz = np.max(arr.shape) - np.array(arr.shape)
        arr = np.pad(arr, ((padx, 0), (pady, 0), (padz, 0)))
    return arr


def rotate_peaks(prep_obj, data, B_recip, voxel_size):
    """Rotates the diffraction pattern of a given peak to the common reference frame"""
    print("rotating diffraction pattern")
    vx_dims = np.array([np.linalg.norm(B_recip[:, i]) for i in range(3)])
    vx_dims = vx_dims / vx_dims.max()
    data = transform.rescale(data, 1/vx_dims, order=5)
    mask = np.ones_like(data)
    data = pad_to_cube(data)
    mask = pad_to_cube(mask)
    print(mask.shape)

    for i in range(3):
        B_recip[:, i] = B_recip[:, i] * vx_dims[i]

    matrix = voxel_size*np.linalg.inv(B_recip)
    center = np.array(data.shape) // 2
    translation = center - np.dot(matrix, center)
    data = ndi.affine_transform(data, matrix, order=5, offset=translation)
    mask = ndi.affine_transform(mask, matrix, order=1, offset=translation)
    mask[mask < 0.99] = 0

    final_size = prep_obj.final_size
    shp = np.array([final_size, final_size, final_size]) // 2

    # Pad the array to the largest dimensions
    shp1 = np.array(data.shape) // 2
    pad = shp - shp1
    pad[pad < 0] = 0
    data = np.pad(data, [(pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2])])
    mask = np.pad(mask, [(pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2])])

    # Crop the array to the final dimensions
    shp1 = np.array(data.shape) // 2
    start, end = shp1 - shp, shp1 + shp
    data = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

    return data, mask.astype("?")


def refine_mask(init_mask, data):
    matrix = 0.8 * np.identity(3)
    center = np.array(data.shape) / 2
    offset = center - np.dot(matrix, center)
    mask = ndi.affine_transform(data, matrix, offset=offset, order=3)
    mask = ndi.gaussian_filter(mask, sigma=5) > 2

    dd = 5
    struct = np.zeros((dd, dd, dd))
    x, y, z = np.mgrid[-1:1:1j*dd, -1:1:1j*dd, -1:1:1j*dd]
    struct[x**2 + y**2 + z**2 < 1] = 1
    mask = ndi.binary_dilation(mask, structure=struct, iterations=1)
    return init_mask | np.invert(mask)


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
            shape = self.prep_obj.read_scan(batch[0][0]).shape
            B_recip, rs_voxel_size = calc_geometry(self.prep_obj, batch[1], shape, self.o_twin)
            batch.append(B_recip)
            batch.append(rs_voxel_size)  # reciprocal-space voxel size in inverse nanometers
            batch.append(2*np.pi/(rs_voxel_size*shape[0]))  # direct-space voxel size in nanometers
        return batches

    def prepare(self, batches):
        processes = []
        # The maximum voxel size in reciprocal space should guarantee the highest resultion in direct space
        rs_voxel_size = max(batch[4] for batch in batches)
        ds_voxel_size = min(batch[5] for batch in batches)
        f = self.prep_obj.experiment_dir + '/conf/config_mp'
        mp_config = ut.read_config(f)
        mp_config["rs_voxel_size"] = rs_voxel_size
        mp_config["ds_voxel_size"] = ds_voxel_size
        ut.write_config(mp_config, f)
        for batch in batches:
            dirs = batch[0]
            scans = batch[1]
            order = batch[2]
            B_recip = batch[3]
            conf_scans = f"{self.prep_obj.scan_ranges[order][0]}-{self.prep_obj.scan_ranges[order][1]}"
            orientation = self.prep_obj.orientations[order]
            orientation = "".join(f"{o}" for o in orientation)
            save_dir = f"{self.prep_obj.experiment_dir}/mp_{conf_scans}_{orientation}/preprocessed_data"
            p = Process(target=self.process_batch,
                        args=(dirs, scans, B_recip, rs_voxel_size, save_dir, 'prep_data.tif'))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def process_batch(self, dirs, scans, B_recip, voxel_size, save_dir, filename):
        batch_arr = combine_scans(self.prep_obj, dirs, scans)
        batch_arr = self.prep_obj.det_obj.clear_seam(batch_arr)
        data, mask = rotate_peaks(self.prep_obj, batch_arr, B_recip, voxel_size)
        mask = refine_mask(mask, data)
        write_prep_arr(data, save_dir, filename)
        write_prep_arr(mask, save_dir, "mask.tif")


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
        for j in range(10):
            image[j] = np.roll(image[j], int(shape[i] / 2) - max_coordinates[i], i)
        support = np.roll(support, int(shape[i] / 2) - max_coordinates[i], i)

    com = ndi.center_of_mass(density * support)
    # place center of mass in the center
    for i in range(len(shape)):
        for j in range(10):
            image[j] = np.roll(image[j], int(shape[i] / 2 - com[i]), axis=i)
        support = np.roll(support, int(shape[i] / 2 - com[i]), axis=i)

    # set center displacement to zero, use as a reference
    half = np.array(shape) // 2
    for i in [1, 2, 3]:
        image[i] = image[i] - image[i, half[0], half[1], half[2]]

    return image, support


def write_vti(data, px, savedir, is_twin=False):
    # Create the vtk object for the data
    if is_twin:
        prepend = "twin_"
    else:
        prepend = ""
    print("Preparing VTK data")
    grid = tvtk.ImageData(dimensions=data[0].shape, spacing=(px, px, px))
    # Set the data to the image/support/distortion
    names = ["density", "u_x", "u_y", "u_z", "s_xx", "s_yy", "s_zz", "s_xy", "s_yz", "s_zx", "support"]
    for img, name in zip(data, names):
        arr = tvtk.DoubleArray()
        arr.from_array(img.ravel())
        arr.name = name
        grid.point_data.add_array(arr)

    # print("Saving VTK")
    # Create the writer object
    writer = tvtk.XMLImageDataWriter(file_name=f"{savedir}/{prepend}full_data.vti")
    writer.set_input_data(grid)
    # Save the data
    writer.write()
    print(f"saved file: {savedir}/{prepend}full_data.vti")


def process_dir(exp_dir, rampups=1, make_twin=True):
    """
    Loads arrays from files in results directory. If reciprocal array exists, it will save reciprocal info in tif
    format.

    Parameters
    ----------
    exp_dir : str
        the directory where phasing results are saved
    rampups : int
        factor to apply to rampups operation, i.e. smoothing the image
    make_twin : bool
        if True visualize twin
    """
    res_dir = Path(exp_dir) / "results_phasing"
    save_dir = Path(exp_dir) / "results_viz"
    # create dir if does not exist
    print(save_dir)
    if not save_dir.exists():
        save_dir.mkdir()
    for f in save_dir.iterdir():
        f.unlink()

    image = np.load(f"{res_dir}/reconstruction.npy")
    image = np.moveaxis(image, 3, 0)
    image[0] = image[0] / np.max(image[0])
    support = np.load(f"{res_dir}/support.npy")

    image, support = center_mp(image, support)
    if rampups > 1:
        image = ut.remove_ramp(image, ups=rampups)

    px = ut.read_config(f"{exp_dir}/conf/config_mp")["ds_voxel_size"]

    write_vti(image, px, save_dir)

    if make_twin:
        image = np.flip(image, axis=(1, 2, 3))
        image[1:-1] *= -1
        if support is not None:
            support = np.flip(support)
            image, support = center_mp(image, support)
        if rampups > 1:
            image = ut.remove_ramp(image, ups=rampups)
        write_vti(image, px, save_dir, is_twin=True)
