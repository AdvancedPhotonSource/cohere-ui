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

def diet_data(dfile):
    gap = 3
    dta = ut.read_tif(dfile)
    shape = dta.shape

    dta = np.take(dta, range(0, shape[-1], gap), axis=2)

    ut.save_tif(dta, dfile)


def tedata(exp_dir):
    fill_ratio = 3
    dfiles = []
    for scan_dir in os.listdir(exp_dir):
        if scan_dir.startswith('scan'):
            dfiles.append(ut.join(exp_dir, scan_dir, 'preprocessed_data', 'prep_data.tif'))

    for i in range(len(dfiles)):
        if i != len(dfiles) - 1 and i % fill_ratio != 0:
            diet_data(dfiles[i])

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
