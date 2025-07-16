# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import argparse
import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab


def xsection(rho, ux, uy, uz, axis=0, contour=0.5):
    ctr = rho.shape[0] // 2
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.7), layout="constrained", sharex='all', sharey='all')
    axs = list(axs[0]) + list(axs[1])

    data = []
    for u in (rho, ux, uy, uz):
        for _ in range(axis):
            u = np.moveaxis(u, 0, -1)
        data.append(u[ctr])
    data[0] = data[0] / np.max(data[0])
    ulimits = np.max([np.quantile(u[u!=0], 0.99)-np.quantile(u[u!=0], 0.01) for u in data[1:]]) / 2
    abcd = ("(a)", "(b)", "(c)", "(d)")
    locations = ("left", "right", "left", "right")
    cmaps = ("gray_r", "seismic", "seismic", "seismic")

    for ax, u, txt, cmap, loc in zip(axs, data, abcd, cmaps, locations):
        print(u.shape)
        if cmap == "gray_r":
            clim = (0, 1)
            contour_color = 'r'
        else:
            u = u - np.mean([np.quantile(u, 0.02), np.quantile(u, 0.98)])
            u = data[0] * u
            clim = (-ulimits, ulimits)
            contour_color = 'k'

        img = ax.imshow(u, cmap=cmap, clim=clim)

        ax.contour(data[0], [contour], colors=contour_color)

        ax.text(0.06, 0.92, txt, transform=ax.transAxes, fontsize=16, horizontalalignment="center",
                verticalalignment="center")
        plt.setp(ax, xticks=[], yticks=[])
        plt.colorbar(img, ax=ax, location=loc)


def show_reconstruction(working_dir, contour=0.5):
    rec = np.load(f"{working_dir}/results_phasing/image_nostrain.npy")
    rho, ux, uy, uz = rec

    mlab.figure()
    mlab.contour3d(rho, color=(1.0, 0.98, 0.94), contours=[contour*np.max(rho)])
    mlab.show()

    for i in range(3):
        xsection(rho, ux, uy, uz, i, contour=contour)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", help="experiment directory")
    args = parser.parse_args()
    show_reconstruction(args.experiment_dir, contour=0.4)
