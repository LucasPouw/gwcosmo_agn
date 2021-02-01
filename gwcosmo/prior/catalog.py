"""
Module containing functionality for creation and management of galaxy catalogs.
Ignacio Magana
"""
import gwcosmo
import numpy as np
import healpy as hp
import pandas as pd

import h5py
import pkg_resources
import time
import progressbar

from ..utilities.standard_cosmology import *
from ..utilities.schechter_function import *

# Global
catalog_data_path = pkg_resources.resource_filename('gwcosmo',
                                                    'data/catalog_data/')

Kcorr_bands = {'B':'B', 'K':'K', 'u':'r', 'g':'r', 'r':'g', 'i':'g', 'z':'r'}
Kcorr_signs = {'B':1, 'K':1, 'u':1, 'g':1, 'r':-1, 'i':-1, 'z':-1}
color_names = {'B':None, 'K':None, 'u':'u - r', 'g':'g - r', 'r':'g - r', 'i':'g - i', 'z':'r - z'}
color_limits = {'u - r':[-0.1,2.9], 'g - r':[-0.1,1.9], 'g - i':[0,3], 'r - z':[0,1.5]}


class galaxy(object):
    """
    Class to store galaxy objects.

    Parameters
    ----------
    index : galaxy index
    ra : Right ascension in radians
    dec : Declination in radians
    z : Galaxy redshift
    m : Apparent magnitude dictonary
    sigmaz : Galaxy redshift uncertainty
    """
    def __init__(self, index=0, ra=0, dec=0, z=1, m={'band':20}, sigmaz=0, Kcorr=0):
        self.index = index
        self.ra = ra
        self.dec = dec
        self.z = z
        self.m = m
        self.Kcorr = Kcorr
        self.sigmaz = sigmaz
        self.dl = self.luminosity_distance()
        self.M = self.absolute_magnitude()
        self.L = self.luminosity()

    def luminosity_distance(self, H0=70.):
        return dl_zH0(self.z, H0)
    
    def absolute_magnitude(self, H0=70., band=None):
        if band is not None:
            return M_mdl(self.m[band], self.dl)
        else:
            M = {}
            for band in self.m:
                M[band] = M_mdl(self.m[band], self.dl)
            return M

    def luminosity(self, band=None):
        if band is not None:
            return L_M(self.M[band])
        else:
            L = {}
            for band in self.M:
                L[band] = L_M(self.M[band])
            return L
 

class galaxyCatalog(object):
    """
    Galaxy catalog class stores a dictionary of galaxy objects.

    Parameters
    ----------
    catalog_file : Path to catalog.hdf5 file
    skymap : gwcosmo.likelihood.skymap.skymap object
    thresh : Probablity contained within sky map region
    band : key of band
    Kcorr : bool, should K corrections be applied?
    nside : resolution of skymap to use for calculating Omega_G and p(x|Omega_G)
    """
    def __init__(self, catalog_file=None, skymap=None, thresh=.9999, band='B', Kcorr=False, nside=64):
        if catalog_file is not None:
            self.catalog_file = catalog_file
            self.dictionary = self.__load_catalog()
            self.skymap = skymap
            self.nside = nside
            
            self.extract_galaxies(skymap=skymap, thresh=thresh, band=band, Kcorr=Kcorr)
            
            if skymap is None:
                self.OmegaG = 1.0
                self.px_OmegaG = 1.0
            else:
                self.OmegaG,self.px_OmegaG = skymap.region_with_sample_support(self.ra, self.dec, thresh, nside=self.nside)

        if catalog_file is None:
            self.catalog_name = ""
            self.dictionary = {'ra': [], 'dec': [], 'z': [], 'sigmaz': [],
                               'skymap_indices': [], 'radec_lim': [], 'm_{0}'.format(band): []}
                               


    def __load_catalog(self):
        return h5py.File(self.catalog_file,'r')

    def nGal(self):
        return len(self.dictionary['z'])

    def get_galaxy(self, index):
        return galaxy(index, self.ra[index], self.dec[index],
                      self.z[index], {self.band: self.m[index]},
                      self.sigmaz[index])

    def mth(self):
        m = self.m
        if sum(m) == 0:
            mth = 25.0
        else:
            mth = np.median(m)
        return mth

    def extract_galaxies(self, skymap=None, thresh=.9, band='B', Kcorr=False):
        band_key = 'm_{0}'.format(band)
        if Kcorr is True:
            band_Kcorr = Kcorr_bands[band]
            band_Kcorr_key = 'm_{0}'.format(band_Kcorr)
            self.color_name = color_names[band]
            self.color_limit = color_limits[self.color_name]
            
        ra = self.dictionary['ra'][:]
        dec = self.dictionary['dec'][:]
        if skymap is None:
            z = self.dictionary['z'][:]
            sigmaz = self.dictionary['sigmaz'][:]
            m = self.dictionary[band_key][:]
            if Kcorr is True:
                m_K = self.dictionary[band_Kcorr_key][:]
            radec_lim = self.dictionary['radec_lim'][:]
        else:
            ind = skymap.samples_within_region(self.dictionary['ra'][:],self.dictionary['dec'][:], thresh, nside=self.nside)
            ra = self.dictionary['ra'][ind]
            dec = self.dictionary['dec'][ind]
            z = self.dictionary['z'][ind]
            sigmaz = self.dictionary['sigmaz'][ind]
            m = self.dictionary[band_key][ind]
            if Kcorr is True:
                m_K = self.dictionary[band_Kcorr_key][ind]
            radec_lim = self.dictionary['radec_lim'][:]
        self.band = band
        self.radec_lim = radec_lim
        if Kcorr is True:
            mask = (~np.isnan(m))&(~np.isnan(m_K))
            m_K = m_K[mask]
        else:
            mask = ~np.isnan(m)
        ra, dec, z, m, sigmaz = ra[mask], dec[mask], z[mask], m[mask], sigmaz[mask]
        self.ra, self.dec, self.z, self.sigmaz, self.m = ra, dec, z, sigmaz, m
        if Kcorr is True:
            self.color = Kcorr_signs[band]*(m - m_K)
        else:
            self.color = np.zeros(len(m))
        return ra, dec, z, m, sigmaz


