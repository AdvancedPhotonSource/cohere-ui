# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import cohere_core.utilities as ut
import argparse
import os
import numpy as np

"""
This script is written to simulate data of time evolving experiment type from multiple scan data taken
for the same sample. It is used for testing only.

Before this script is run, the beamline_preprocessing must be completed with the separate_scans set to True,
so the prep_data.tif data files are generated in preprocessed_data subdirectory of each scan directory.
This script removes frames from the scans between 'gap' scans. The 'gap' scans contain full data. The gap
is set to 3.
The frames are removed with the rato equal to fill_ratio variable, set to 3.
"""

def tedata(exp_dir):
    fill_ratio = 3
    gap = 3
    dfiles = []
    for scan_dir in os.listdir(exp_dir):
        if scan_dir.startswith('scan'):
            dfiles.append(ut.join(exp_dir, scan_dir, 'preprocessed_data', 'prep_data.tif'))

    # read the first file and place the max in the slice that is preserved, i.e. it divides by gap
    dta = ut.read_tif(dfiles[0])
    third_dim = dta.shape[-1]
    dta, _ = ut.center_max(dta)
    max_ind = np.unravel_index(np.argmax(dta), dta.shape)
    max_slice_inx = max_ind[-1]

    good_dim = ut.get_good_dim(third_dim, 'cp')
    pad = 0

    while (max_slice_inx + pad) % gap != 0:
        pad += 1
    # if padding added to dim is greater than dim, then add only to the good dim and roll
    if third_dim + pad > good_dim:
        roll = third_dim + pad - good_dim
        pad = good_dim - third_dim
    else:
        roll = pad
        pad = 0
    if pad > 0:
        pad_width = ((0, 0), (0, 0), (pad, 0))
        dta = np.pad(dta, pad_width, mode='constant', constant_values=0)
    if roll != 0:
        dta = np.roll(dta, roll, axis=2)
    ut.save_tif(dta, dfiles[0])
    max_slice_inx = max_slice_inx + pad + roll
    third_dim = third_dim + pad

    for i in range(1, len(dfiles)):
        # align max with the first scan data
        dta = ut.read_tif(dfiles[i])
        dta, max_ind = ut.center_max(dta)
        # assuming the max is at the same index as the first scan
        if pad > 0:
            pad_width = ((0, 0), (0, 0), (pad, 0))
            dta = np.pad(dta, pad_width, mode='constant', constant_values=0)
        if roll != 0:
            dta = np.roll(dta, roll, axis=2)

        if i != len(dfiles) - 1 and i % fill_ratio != 0:
            dta = np.take(dta, range(0, dta.shape[-1], gap), axis=2)
        ut.save_tif(dta, dfiles[i])

    for f in dfiles:
        d = ut.read_tif(f)
        print('shape', d.shape, np.unravel_index(np.argmax(d), d.shape), np.max(d))


def main():
        parser = argparse.ArgumentParser()
        parser.add_argument("exp_dir", help="experiment dir.")
        args = parser.parse_args()
        tedata(args.exp_dir)


if __name__ == "__main__":
    exit(main())
