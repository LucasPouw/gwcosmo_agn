#!/usr/bin/env python3

"""
Compute the coarse mth map for generating the LOS redshift prior

Freija Beirnaert
"""

import numpy as np
import healpy as hp
import sys
from tqdm import tqdm

import gwcosmo

import logging
handler_out = logging.StreamHandler(stream=sys.stdout)
handler_err = logging.StreamHandler(stream=sys.stderr)
handler_err.setLevel(logging.ERROR)
logging.basicConfig(handlers=[handler_out, handler_err], level = logging.INFO)
logger = logging.getLogger(__name__)

def create_mth_map(outfile, catalog, band, nside, min_gal, pixel_index):
    """
    Generates a healpix map containing a magnitude threshold 
    based on the galaxies of a particular pixel in the sky.
    The result is saved as a fits file.

    Parameters
    ----------
    outfile : str
        path to the output file
    catalog : str
        galaxy catalogue
    band : str
        bbservation band of galaxy catalog (B,K,W1,bJ,u,g,r,i,z)
    nside : int
        healpy skymap nside
    min_gal : int
        the minimum number of galaxies in a pixel 
        for it not to be considered empty
    pixel_index: int
        healpy skymap pixel index for which mth is calculated
        if None, a full skymap is generated

    Returns
    -------
    None
    """
    
    logger.info(f"create {outfile}")
    full_catalog = gwcosmo.prior.catalog.load_catalog(catalog, band)

    npix = hp.pixelfunc.nside2npix(int(nside))
    m = np.zeros(npix)

    ipix = [pixel_index] # one pixel
    if pixel_index == None:
        ipix = np.arange(0, npix, 1) # full sky

    for pix in tqdm(ipix):
        subcatalog = full_catalog.select_pixel(nside, pix, nested=True)
        mth = subcatalog.magnitude_thresh(band,min_gals=int(min_gal))
        m[pix] = mth

    hp.fitsfunc.write_map(outfile, m, nest=True, overwrite=True)

