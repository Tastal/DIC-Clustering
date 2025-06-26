# -*- coding: utf-8 -*-
"""Load ECCO-Darwin DIC data.

This module is a direct conversion of the Jupyter notebook
``Read_model_file.ipynb``. It demonstrates how to read the binary
``.data``/``.meta`` files from an llc270 ECCO model output, unscramble the
13 tiles into a regular matrix and interpolate the surface field to a
regular latitude/longitude grid using ``ecco_v4_py``. The functions and
variables mirror those used in the original notebook.
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import warnings
from pprint import pprint
import importlib
from matplotlib.gridspec import GridSpec
import matplotlib as mpl

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Helper functions from the notebook
# -----------------------------------------------------------------------------

def decode_llc(fname, nTr):
    """Decode an llc270 binary file."""
    nX = 270
    nY = nX * 13
    nZ = 50
    with open(fname, 'rb') as fid:
        tmp = np.fromfile(fid, '>f4')
    fld = tmp.reshape((nTr, nZ, nY, nX))
    return fld


def plot_tiles(data, tsz):
    """Plot the 13 llc tiles in their native layout."""
    iid = [4, 3, 2, 4, 3, 2, 1, 1, 1, 1, 0, 0, 0]
    jid = [0, 0, 0, 1, 1, 1, 1, 2, 3, 4, 2, 3, 4]
    tid = 0
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(5, 5, wspace=.05, hspace=.05)
    for i in range(len(iid)):
        ax = fig.add_subplot(gs[iid[i], jid[i]])
        if i >= 7:  # tiles 8--13 need to be rotated
            ax.imshow(data[tid:tid + tsz].T, origin='lower')
        else:
            ax.imshow(data[tid:tid + tsz], origin='lower')
        tid += tsz
        ax.tick_params(bottom=False, labelbottom=False,
                       left=False, labelleft=False)
    plt.show()


def transp_tiles(data):
    """Unscramble tiles that are stored with flipped axes."""
    nx = data.shape[1]
    tmp = data[7 * nx:, ::-1]
    transpo = np.concatenate((
        tmp[2::3, :].transpose(),
        tmp[1::3, :].transpose(),
        tmp[0::3, :].transpose()))
    data_out = np.concatenate((data[:7 * nx],
                               np.flipud(transpo[:, :nx]),
                               np.flipud(transpo[:, nx:])))
    return data_out


def decode_llc_grid(fname, ndim):
    """Decode grid variables."""
    nX = 270
    nY = nX * 13
    nZ = 50
    with open(fname, 'rb') as fid:
        tmp = np.fromfile(fid, '>f4')
    if ndim == 2:
        fld = tmp.reshape((nY, nX))
    else:
        fld = tmp.reshape((nZ,))
    return fld


# -----------------------------------------------------------------------------
# Demonstration of loading the model output
# -----------------------------------------------------------------------------

def main():
    """Replicate the workflow of the original notebook."""
    filename = "average_DIC_3d.0000002232.data"
    num_variables = 15  # 15 variables contained in the file
    data_raw = decode_llc(filename, num_variables)

    # Display dimensions of the loaded array
    print("raw shape", data_raw.shape)

    # Extract the DIC variable
    data_dic = data_raw[0]
    print("dic shape", data_dic.shape)

    # Unscramble tiles 8--13 at each depth level
    for ind in range(data_dic.shape[0]):
        data_dic[ind] = transp_tiles(data_dic[ind])

    # Mask out land and sea floor (set to NaN)
    data_dic[data_dic == 0] = np.nan

    # Plot the surface DIC concentration in native tile layout
    Nx = 270
    plot_tiles(data_dic[0], Nx)

    # Read grid information and unscramble
    xc = decode_llc_grid('XC.data', 2)
    xc = transp_tiles(xc)
    yc = decode_llc_grid('YC.data', 2)
    yc = transp_tiles(yc)
    z = decode_llc_grid('RC.data', 1)

    print(xc.shape, yc.shape, z.shape)

    # Plot vertical profiles at random locations
    fig, ax = plt.subplots()
    ax.plot(data_dic[:, np.random.randint(0, 3509), np.random.randint(0, 269)], z)
    ax.plot(data_dic[:, np.random.randint(0, 3509), np.random.randint(0, 269)], z)
    ax.plot(data_dic[:, np.random.randint(0, 3509), np.random.randint(0, 269)], z)
    ax.plot(data_dic[:, np.random.randint(0, 3509), np.random.randint(0, 269)], z)
    ax.set(xlabel='DIC concentration', ylabel='Depth $z$ (m)')
    plt.show()

    # Interpolate the surface field onto a regular 1x1 degree grid
    from os.path import join, expanduser
    user_home_dir = expanduser('~')
    sys.path.append(join(user_home_dir, 'ECCOv4-py'))
    import ecco_v4_py as ecco

    new_grid_delta_lat = 1
    new_grid_delta_lon = 1
    new_grid_min_lat = -90
    new_grid_max_lat = 90
    new_grid_min_lon = -180
    new_grid_max_lon = 180

    new_grid_lon_centers, new_grid_lat_centers, \
        new_grid_lon_edges, new_grid_lat_edges, \
        dic_interp = ecco.resample_to_latlon(
            xc,
            yc,
            data_dic[0],
            new_grid_min_lat, new_grid_max_lat, new_grid_delta_lat,
            new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,
            fill_value=np.NaN,
            mapping_method='nearest_neighbor',
            radius_of_influence=120000)

    print('interpolated shape', dic_interp.shape)

    lat_interp = np.linspace(-90, 90, 180)
    lon_interp = np.linspace(-180, 180, 360)

    fig, ax = plt.subplots()
    im = ax.pcolor(lon_interp, lat_interp, dic_interp, vmin=1700, vmax=2200)
    plt.colorbar(im, ax=ax)
    ax.set(xlabel='longitude', ylabel='latitude',
           title='surface DIC concentration')
    plt.show()


if __name__ == '__main__':
    main()
