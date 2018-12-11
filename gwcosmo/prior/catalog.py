"""Module containing functionality for creation and management of galaxy catalogs.
Ignacio Magana
"""

import numpy as np
import healpy as hp
import pandas as pd

from ligo.skymap.bayestar import rasterize
from ligo.skymap.moc import nest2uniq,uniq2nest

import pickle
import pkg_resources

# Global
catalog_data_path = pkg_resources.resource_filename('gwcosmo', 'data/catalog_data/')

class galaxy(object):
    """
    Class to store galaxy objects.
    
    Parameters
    ----------
    index : galaxy index 
    ra : Right ascension in radians
    dec : Declination in radians
    z : redshift
    m : Apparent blue magnitude
    sigmaz : redshift uncertainty
    """
    def __init__(self, index = 0, ra = 0, dec = 0, z = 0, m = 0, sigmaz = 0):
        self.index = index
        self.ra = ra
        self.dec = dec
        self.z = z
        self.m = m
        self.sigmaz = sigmaz

    def blue_luminosity_from_mag(self):
        """
        Returns the blue luminosity in units of L_10 given the apparent
        magnitude and the redshift of a galaxy object assuming a Hubble constant of 70.
        """
        coverh = (const.c.to('km/s') / ( 70 * u.km / u.s / u.Mpc )).value
        M_blue_solar = 5.48 # Binney & Tremaine
        MB = self.m - 5.0 * np.log10( self.z*coverh / 10.0e-6 )
        lumB = np.power( 10, (M_blue_solar - MB)/2.5 - 10.0 ) 
        return lumB
        
class galaxyCatalog(object):
    """
    Galaxy catalog class stores a dictionary of galaxy objects.
    
    Parameters
    ----------
    catalog_file : Path to catalog.p file
    catalog_name : Name of stored catalog to be loaded
    """
    def __init__(self, catalog_file=None, catalog_name=None):
        if catalog_file is not None:
            self.catalog_file = catalog_file
            self.dictionary = self.__load_catalog()
        if catalog_name is not None:
            self.catalog_name = catalog_name
            self.dictionary = self.__load_stored_catalog()
        if (catalog_file is None and catalog_name is None):
            print("Making empty catalog")
            self.catalog_name = ""
            self.dictionary = {'0':galaxy()}

    def __load_catalog(self):
        return pickle.load(open(self.catalog_file, "rb"))

    def nGal(self):
        return len(self.dictionary)

    def get_galaxy(self,index):
        return self.dictionary[str(int(index))]
    
    def mth(self):
        ngal = self.nGal()
        m = np.zeros(ngal)
        for i in range(ngal):
            m[i] = self.get_galaxy(i).m
        if sum(m) == 0:
            mth = 25.0
        else:
            mth = np.median(m)
        return mth

    def extract_galaxies(self):
        nGal = self.nGal()
        ra = np.zeros(nGal)
        dec = np.zeros(nGal)
        z = np.zeros(nGal)
        m = np.zeros(nGal)
        sigmaz = np.zeros(nGal)
        for i in range(nGal):
            gal = self.get_galaxy(i)
            ra[i] = gal.ra
            dec[i] = gal.dec
            z[i] = gal.z
            m[i] = gal.m
            sigmaz[i] = gal.sigmaz
        if all(m) == 0: #for mdc1 and mdc2
            m = np.ones(nGal)
        return ra, dec, z, m, sigmaz

    def pixelCatalogs(self,skymap3d):
        moc_map = skymap3d.as_healpix()
        pixelmap = rasterize(moc_map)
        nside = hp.npix2nside(len(pixelmap))
        # For each pixel in the sky map, build a list of galaxies and the effective magnitude threshold
        self.pixel_cats = {} # dict of list of galaxies, indexed by NUNIQ
        ra, dec, _, _, _ = self.extract_galaxies()
        gal_idx = np.array(range(len(ra)),dtype=np.uint64)
        theta = np.pi/2.0 - dec
        # Find the NEST index for each galaxy
        pix_idx = np.array(hp.ang2pix(nside, theta, ra,nest=True),dtype=np.uint64)
        # Find the order and nest_idx for each pixel in the UNIQ map
        order, nest_idx = uniq2nest(moc_map['UNIQ'])
        # For each order present in the UNIQ map
        for o in np.unique(order):
            # Find the UNIQ pixel for each galaxy at this order
            uniq_idx = nest2uniq(o, pix_idx)
            # For each such pixel, if it is in the moc_map then store the galaxies
            for uidx in moc_map['UNIQ'][order==o]:
                self.pixel_cats[uidx]=[self.get_galaxy(j) \
                                           for j in gal_idx[uniq_idx==uidx]]
        return self.pixel_cats