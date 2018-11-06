"""Module containing functionality for creation and management of galaxy catalogs.
Ignacio Magana
"""

import numpy as np
from astropy.table import Table
from scipy.stats import gaussian_kde
import pandas as pd
from astropy.io import fits
from astropy import constants as const
from astropy import units as u

import pkg_resources

# Global
catalog_data_path = pkg_resources.resource_filename('gwcosmo', 'data/catalog_data/')

def blue_luminosity_from_mag(m,z):
    """
    Returns the blue luminosity in units of L_10 given the apparent
    magnitude and the luminosity distance
    """
    coverh = (const.c.to('km/s') / ( 70 * u.km / u.s / u.Mpc )).value
    M_blue_solar = 5.48 # Binney & Tremaine
    MB = m - 5.0 * np.log10( z*coverh / 10.0e-6 )
    lumB = np.power( 10, (M_blue_solar - MB)/2.5 - 10.0 ) 
    return lumB

class galaxy(object):
    ''' Class for galaxy objects
    '''
    def __init__(self, index = 0, ra = 0, dec = 0, z = 0, m = 0, lumB = 1.0):
        """Galaxy catalog class... 
        Parameters
        """
        self.index = index
        self.ra = ra
        self.dec = dec
        self.z = z
        self.m = m
        self.lumB = lumB
        
    def load_astropy_row_glade(self, index, row):
        self.index = index
        self.ra = row['RA']*np.pi/180.
        self.dec = row['Dec']*np.pi/180.
        self.z = row['z']
        self.m = row['Bmag']
        self.lumB = row['lumB']

    def load_astropy_row_mdc(self, index, row, version):
        self.index = index
        self.ra = row['RA']
        self.dec = row['Dec']
        self.z = row['z']
        if version != "1.0":
            self.m = row['m']
            self.lumB = blue_luminosity_from_mag(self.m,self.z)
            
    def load_counterpart(self, ra, dec, z):
        self.index = 0
        self.ra = ra
        self.dec = dec
        self.z = z
        
    def load_row_mice(self, index, row):
        self.index = index
        self.ra = row.ra_gal*np.pi/180.
        self.dec = row.dec_gal*np.pi/180.
        self.z = row.z_cgal_v 
        self.m = row.des_asahi_full_r_true
        self.lumB = blue_luminosity_from_mag(self.m,self.z)

    def load_astropy_row_sdss_cluster(self, index, row):
        self.index = index
        self.ra = row['RA']
        self.dec = row['Dec']
        self.m = row['rmag']
        if row['zspec'] == -1.0:
            self.z = row['zphoto']
        else:
            self.z = row['zspec']
        self.lumB = blue_luminosity_from_mag(self.m,self.z)
        
    def load_astropy_row_sdss(self, index, row):
            self.index = index
            self.ra = row['RA']*np.pi/180.
            self.dec = row['Dec']*np.pi/180.
            self.z = row['z']
            self.m = row['abs_mag_r']
            self.lumB = blue_luminosity_from_mag(self.m,self.z)
        
class galaxyCatalog(object):
    ''' Class for galaxy catalog objects
    '''
    def __init__(self, catalog_file = "", indexes=0, dictionary={}):
        """Galaxy catalog class... 
        Parameters
        """
        self.catalog_file = catalog_file
        self.indexes = indexes
        self.dictionary = dictionary
        
    def load_counterpart_catalog(self, ra, dec, z):   
        galaxies={}
        nGal = 1
        gal = galaxy()
        gal.load_counterpart(ra, dec, z)
        galaxies[str(0)] = gal
        self.dictionary = galaxies
        self.indexes = np.arange(nGal)
        
    def load_glade_catalog(self):
        self.catalog_file = catalog_data_path + "gladecatalogv2.3_corrected.dat"
        t = Table.read(self.catalog_file,format='ascii')
        galaxies={}
        nGal = len(t)
        for k in range(0,nGal):
            gal = galaxy()
            gal.load_astropy_row_glade(k,t[k])
            galaxies[str(k)]= gal
        self.dictionary = galaxies
        self.indexes = np.arange(nGal)

    def load_mdc_catalog(self,version='1.0'):
        if version == '1.0':
            self.catalog_file = catalog_data_path + "mdc_v1_cat.txt"
        if version == '2.1':
            self.catalog_file = catalog_data_path + "mdc_v2_lim_cat.txt"
        if version == '2.2':
            self.catalog_file = catalog_data_path + "mdc_v2-2_lim_cat.txt"
        if version == '2.3':
            self.catalog_file = catalog_data_path + "mdc_v2-3_lim_cat.txt"            
        if version == '3.1':
            self.catalog_file = catalog_data_path + "mdc_v3_lim_cat.txt"

        t = Table.read(self.catalog_file,format='ascii')
        galaxies={}
        nGal = len(t)
        for k in range(0,nGal):
            gal = galaxy()
            gal.load_astropy_row_mdc(k,t[k],version)
            galaxies[str(k)]= gal
        self.dictionary = galaxies
        self.indexes = np.arange(nGal)
        
    def load_mice_catalog(self):
        "MICE catalog: /home/ignacio.magana/src/gwcosmo/gwcosmo/data/catalog_data/mice.fits"
        self.catalog_file = catalog_data_path + "gladecatalogv2.3_corrected.dat"
        with fits.open('mice_test.fits') as data:
            df = pd.DataFrame(np.array(data[1].data).byteswap().newbyteorder())

        galaxies={}
        nGal = len(df)
        for k in range(0,nGal):
            gal = galaxy()
            gal.load_row_mice(k,df.iloc[k])
            galaxies[str(k)]= gal
        self.dictionary = galaxies
        self.indexes = np.arange(nGal)
    
    def load_SDSS_cluster_catalog(self):
        "/home/ignacio.magana/src/gwcosmo/gwcosmo/data/catalog_data/SDSS170818_clusters.dat"
        self.catalog_file = catalog_data_path + "SDSS170818_clusters.dat"
        t = Table.read(self.catalog_file,format='ascii')

        galaxies={}
        nGal = len(t)
        for k in range(0,nGal):
            gal = galaxy()
            gal.load_astropy_row_sdss_cluster(k,t[k])
            galaxies[str(k)]= gal
        self.dictionary = galaxies
        self.indexes = np.arange(nGal)

    def load_SDSS_catalog(self):
        "/home/ignacio.magana/src/gwcosmo/gwcosmo/data/catalog_data/SDSS170818_BLUE.dat"
        self.catalog_file = catalog_data_path + "SDSS170818_BLUE.dat"
        t = Table.read(self.catalog_file,format='ascii')

        galaxies={}
        nGal = len(t)
        for k in range(0,nGal):
            gal = galaxy()
            gal.load_astropy_row_sdss(k,t[k])
            galaxies[str(k)]= gal
        self.dictionary = galaxies
        self.indexes = np.arange(nGal)
        
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
