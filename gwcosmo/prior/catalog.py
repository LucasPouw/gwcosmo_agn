"""
Module containing functionality for creation and management of galaxy catalogs.
Ignacio Magana
"""
import gwcosmo
import numpy as np
import healpy as hp
import pandas as pd

from ligo.skymap.bayestar import rasterize
from ligo.skymap.moc import nest2uniq, uniq2nest

import pickle
import pkg_resources
import time
import progressbar

# Global
catalog_data_path = pkg_resources.resource_filename('gwcosmo',
                                                    'data/catalog_data/')


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
    def __init__(self, index=0, ra=0, dec=0, z=0, m=0, sigmaz=0):
        self.index = index
        self.ra = ra
        self.dec = dec
        self.z = z
        self.m = m
        self.sigmaz = sigmaz


class galaxyCatalog(object):
    """
    Galaxy catalog class stores a dictionary of galaxy objects.

    Parameters
    ----------
    catalog_file : Path to catalog.p file
    catalog_name : Name of stored catalog to be loaded
    """
    def __init__(self, catalog_file=None, skypatch=None):
        if catalog_file is not None:
            self.catalog_file = catalog_file
            self.dictionary, self.skypatch = self.__load_catalog()
        if catalog_file is None:
            self.catalog_name = ""
            self.dictionary = {'0': galaxy()}
            self.skypatch = None

        self.ra, self.dec, self.z, self.m, self.sigmaz = self.extract_galaxies()

    def __load_catalog(self):
        cat = pickle.load(open(self.catalog_file, "rb"))
        dictionary = cat[0]
        skypatch = cat[1]
        return dictionary, skypatch

    def nGal(self):
        return len(self.dictionary)

    def get_galaxy(self, index):
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
        if all(m) == 0:  # for mdc1 and mdc2
            m = np.ones(nGal)
        return ra, dec, z, m, sigmaz

    def redshiftUncertainty(self, peculiarVelocityCorr=False, nsmear=10):
        """
        A function which "smears" out galaxies in the catalog, therefore
        incorporating redshift uncetainties.
        """
        z_uncert = []
        ralist, declist, zlist, mlist, sigmaz = self.extract_galaxies()
        if peculiarVelocityCorr == True:
            nsmear = 10000
        if peculiarVelocityCorr == False:
            nsmear = nsmear
        zmaxmax = 1.0
        ra_uncert = np.repeat(ralist, nsmear)
        dec_uncert = np.repeat(declist, nsmear)
        m_uncert = np.repeat(mlist, nsmear)
        for i, z in enumerate(zlist):
            z_uncert.append(z+sigmaz[i]*np.random.randn(nsmear))
        z_uncert = np.array(z_uncert).flatten()
        sel = (z_uncert > 0.) & (z_uncert < zmaxmax)
        z_uncert = z_uncert[sel]
        ra_uncert = ra_uncert[sel]
        dec_uncert = dec_uncert[sel]
        m_uncert = m_uncert[sel]

        galaxies = {}
        index = np.arange(len(z_uncert))

        bar = progressbar.ProgressBar()
        print("Loading galaxy catalog.")
        for i in bar(index):
            gal = gwcosmo.prior.catalog.galaxy()
            gal.ra = ra_uncert[i]
            gal.dec = dec_uncert[i]
            gal.z = z_uncert[i]
            gal.m = m_uncert[i]
            gal.sigmaz = 0.
            galaxies[str(i)] = gal
        catalog = gwcosmo.prior.catalog.galaxyCatalog()
        catalog.dictionary = galaxies
        return catalog
