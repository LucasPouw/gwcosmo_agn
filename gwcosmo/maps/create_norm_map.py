#!/usr/bin/env python3

"""
Compute the coarse galaxy_norm map for generating the LOS redshift prior

Freija Beirnaert
"""

import numpy as np
import healpy as hp
import sys

import gwcosmo
from scipy.integrate import quad, dblquad
from scipy.stats import norm, truncnorm
from scipy.interpolate import interp1d
from gwcosmo.utilities.luminosity_function import M_Mobs,M_mdl
from gwcosmo.prior.catalog import color_names, color_limits

import logging
handler_out = logging.StreamHandler(stream=sys.stdout)
handler_err = logging.StreamHandler(stream=sys.stderr)
handler_err.setLevel(logging.ERROR)
logging.basicConfig(handlers=[handler_out, handler_err], level = logging.INFO)
logger = logging.getLogger(__name__)

def calc_norm(catalog, band, nside, pixel_index, zmax, zcut, mth, cosmology, schech_Mmax, apply_Kcorr):
    """
    Calculate the average normalization factor based on the galaxies 
    of a particular healpy pixel in the sky.
    The resulting number is used in the LOS redshift prior calculation.

    Parameters
    ----------
    catalog : str
        galaxy catalogue
    band : str
        bbservation band of galaxy catalog (B,K,W1,bJ,u,g,r,i,z)
    nside : int
        healpy skymap nside
    pixel_index: int
        healpy skymap pixel index for which mth is calculated
        if None, a full skymap is generated
    zmax: float
        upper redshift limit for integrals
    zcut: float
        hard redshift cut to apply to the galaxy catalogue
    mth: float
        value of the magnitude threshold
    cosmology: gwcosmo.utilities.cosmology.standard_cosmology object
        cosmological model
    schech_Mmax : float
        value for Mmax of schechter function for a given band   
    apply_Kcorr: bool
        Apply K-corrections
    
    Returns
    -------
    float
    """

    subcatalog = catalog.select_pixel(nside, pixel_index, nested=True)
    color_limit = 0.
    if apply_Kcorr:
        color_limit = color_limits[color_names[band]]
    else:
        color_limit = [-np.inf,np.inf]
    subcatalog = subcatalog.apply_redshift_cut(zcut).apply_color_limit(band, color_limit[0], color_limit[1]).apply_magnitude_limit(band, mth)

    zs = subcatalog['z'] 
    ms = subcatalog.get_magnitudes(band)
    sigmazs = subcatalog['sigmaz']
    colors = subcatalog.get_color(band)
    Mmax = M_Mobs(cosmology.H0,schech_Mmax)
    
    galaxy_norm = 0.
    N_in3sigma = 0
    logger.info(f"nside {nside} pixel {pixel_index}: The initial number of galaxies in this pixel is {len(zs)}")
    for i in range(len(zs)):
        z_array = np.linspace(zs[i]-5*sigmazs[i],zs[i]+5*sigmazs[i],50) # redshift array for truncnorm interpolations in redshiftuncertainties
        # set redshift limits so that galaxies can't have negative z
        low_z_lim, high_z_lim = (0 - zs[i])/sigmazs[i], (zmax - zs[i])/sigmazs[i]
        # Find the effective number of galaxies below zcut. If zcut=zmax then galaxy_norm=Ngal.
        # Takes into account galaxies whose z distribution crosses zcut.
        if low_z_lim > -3:
            N_in3sigma += 1
        if apply_Kcorr:
            Kcorr = catalog.get_k_correction(band, z_array, color_names[band], colors[i])
        else:
            Kcorr = 0.
        # subtract part where absolute magnitude (if it was located at that z) of the galaxy exceeds Mmax
        subt = 0.
        inds = np.where(M_mdl(ms[i],cosmology.dl_zH0(z_array),Kcorr=Kcorr) > Mmax)[0]
        if inds.size != 0:
            ind = int(inds[-1])
            low_zcut = min(zcut, z_array[ind])
            subt = truncnorm.cdf(low_zcut, low_z_lim, high_z_lim, zs[i], sigmazs[i]) 
        galaxy_norm += truncnorm.cdf(zcut, low_z_lim, high_z_lim, zs[i], sigmazs[i]) - subt

    logger.info(f"nside {nside} pixel {pixel_index}: The effective number of galaxies with z=0 within 3sigma is {N_in3sigma}")
    if len(zs) != len(np.where(zs<zcut)[0]):
        logger.warn(f'The effective number of galaxies below z={zcut} is estimated to be {galaxy_norm}.')

    return galaxy_norm    


def create_norm_map(outfile, catalog, band, nside, pixel_index, zmax, zcut, mth, cosmology, schech_Mmax, apply_Kcorr):
    """
    Generates a healpix map containing an average norm value 
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
    pixel_index: int
        healpy skymap pixel index for which mth is calculated
        if None, a full skymap is generated
    zmax: float
        upper redshift limit for integrals
    zcut: float
        hard redshift cut to apply to the galaxy catalogue
    mth: str
        path to the magnitude threshold map
    cosmology: gwcosmo.utilities.cosmology.standard_cosmology object
        cosmological model
    schech_Mmax : float
        value for Mmax of schechter function for a given band   
    apply_Kcorr: bool
        Apply K-corrections
    
    Returns
    -------
    None
    """

    logger.info(f"create {outfile}")
    galaxy_catalog = gwcosmo.prior.catalog.load_catalog(catalog, band)

    mth_map = hp.fitsfunc.read_map(mth, nest=True)  

    npix = hp.pixelfunc.nside2npix(int(nside))
    m = np.zeros(npix)

    ipix = [pixel_index] # one pixel
    if pixel_index == None:
        ipix = np.arange(0, npix, 1) # full sky

    for pix in ipix:
        mth_val = mth_map[pix]
        galaxy_norm = calc_norm(galaxy_catalog, band, nside, pix, zmax, zcut, mth_val, cosmology, schech_Mmax, apply_Kcorr)
        m[pix] = galaxy_norm

    hp.fitsfunc.write_map(outfile, m, nest=True, overwrite=True)
