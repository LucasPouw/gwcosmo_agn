#!/usr/bin/env python3

import numpy as np
import h5py
import math

def get_offset(LOS_catalog):
    diction = eval(LOS_catalog.attrs['opts'])
    return diction["offset"]

def get_z_array(LOS_catalog):
    return LOS_catalog['z_array'][:]

def get_array(LOS_catalog, arr_name):
    offset = get_offset(LOS_catalog)

    arr = LOS_catalog[str(arr_name)][:]
    arr = np.exp(arr)
    arr -= offset

    return arr

def get_empty_catalog(LOS_catalog):
    return get_array(LOS_catalog, "empty_catalogue")

def get_zprior_full_sky(LOS_catalog):
    return get_array(LOS_catalog, "combined_pixels")

def get_zprior(LOS_catalog, pixel_index):
    return get_array(LOS_catalog, str(pixel_index))

def create_zarray(catalog, band):
    """
    Return zarray for given catalog and band.
    Currently implemented: GLADE+ (B, K, W1, bJ)
    """
    if 'GladePlus' in catalog.name:
        if band in ['B', 'K', 'W1']:
            m = 0.9

            z_lin1 = [1e-6, 4e-5]
            sigmaz1 = 1.5e-6

            z_log1 = [4e-5, 5e-3]
            sigmazoverz1 = 2e-2

            z_lin2 = [5e-3, 1]
            sigmaz2 = 9e-5

            z_log2 = [1, 10]
            sigmazoverz2 = 1e-2/2

            nlin1 = n_linear(z_lin1[0], z_lin1[1], sigmaz1, m)
            nlog1 = n_logarithmic(z_log1[0], z_log1[1], sigmazoverz1, m)
            nlin2 = n_linear(z_lin2[0], z_lin2[1], sigmaz2, m)
            nlog2 = n_logarithmic(z_log2[0], z_log2[1], sigmazoverz2, m)

            zarray = np.linspace(z_lin1[0], z_lin1[1], nlin1, endpoint=False)
            zarray = np.append(zarray, np.logspace(np.log10(z_log1[0]), np.log10(z_log1[1]), nlog1, endpoint=False))
            zarray = np.append(zarray, np.linspace(z_lin2[0], z_lin2[1], nlin2, endpoint=False))
            zarray = np.append(zarray, np.logspace(np.log10(z_log2[0]), np.log10(z_log2[1]), nlog2))

        elif band == 'bJ':
            m = 0.9

            z_lin1 = [1e-6, 5e-4]
            sigmaz1 = 1e-5

            z_log1 = [5e-4, 3e-3]
            sigmazoverz1 = 2e-2

            z_lin2 = [3e-3, 1]
            sigmaz2 = 1e-4

            z_log2 = [1, 10]
            sigmazoverz2 = 0.1

            nlin1 = n_linear(z_lin1[0], z_lin1[1], sigmaz1, m)
            nlog1 = n_logarithmic(z_log1[0], z_log1[1], sigmazoverz1, m)
            nlin2 = n_linear(z_lin2[0], z_lin2[1], sigmaz2, m)
            nlog2 = n_logarithmic(z_log2[0], z_log2[1], sigmazoverz2, m)

            zarray = np.linspace(z_lin1[0], z_lin1[1], nlin1, endpoint=False)
            zarray = np.append(zarray, np.logspace(np.log10(z_log1[0]), np.log10(z_log1[1]), nlog1, endpoint=False))
            zarray = np.append(zarray, np.linspace(z_lin2[0], z_lin2[1], nlin2, endpoint=False))
            zarray = np.append(zarray, np.logspace(np.log10(z_log2[0]), np.log10(z_log2[1]), nlog2))
    else:   #If catalog is not implemented, return zarray for GLADE+ K band.
        m = 0.9

        z_lin1 = [1e-6, 4e-5]
        sigmaz1 = 1.5e-6

        z_log1 = [4e-5, 3e-3]
        sigmazoverz1 = 2e-2

        z_lin2 = [3e-3, 1]
        sigmaz2 = 1e-4

        z_log2 = [1, 10]
        sigmazoverz2 = 1e-2/2

        nlin1 = n_linear(z_lin1[0], z_lin1[1], sigmaz1, m)
        nlog1 = n_logarithmic(z_log1[0], z_log1[1], sigmazoverz1, m)
        nlin2 = n_linear(z_lin2[0], z_lin2[1], sigmaz2, m)
        nlog2 = n_logarithmic(z_log2[0], z_log2[1], sigmazoverz2, m)

        zarray = np.linspace(z_lin1[0], z_lin1[1], nlin1, endpoint=False)
        zarray = np.append(zarray, np.logspace(np.log10(z_log1[0]), np.log10(z_log1[1]), nlog1, endpoint=False))
        zarray = np.append(zarray, np.linspace(z_lin2[0], z_lin2[1], nlin2, endpoint=False))
        zarray = np.append(zarray, np.logspace(np.log10(z_log2[0]), np.log10(z_log2[1]), nlog2))
    return zarray



def n_linear(zmin, zmax, sigmaz, m):
    """
    Returns the number of points between zmin and zmax on the linear scale with the condition that there are  m points (on avarage) between z and z+sigmaz.
    """
    return int((zmax-zmin)/(sigmaz/m))

def n_logarithmic(zmin, zmax, sigmazoverz, m):
    """
    Returns the number of points between zmin and zmax on the logarithmic scale with the condition that there are  m points (on avarage) between z and z+sigma
    """
    return int(m*math.log(zmax/zmin, 1+sigmazoverz))
