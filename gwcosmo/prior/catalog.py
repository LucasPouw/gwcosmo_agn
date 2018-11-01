"""Module containing functionality for creation and management of galaxy catalogs.
Ignacio Magana
"""

import numpy as np
from astropy.table import Table
from scipy.stats import gaussian_kde
import pandas as pd
from astropy.io import fits

import pkg_resources

# Global
catalog_data_path = pkg_resources.resource_filename('gwcosmo', 'data/catalog_data/')

class galaxy(object):
    ''' Class for galaxy objects
    '''
    def __init__(self, index = 0, pgc_number = 0, galaxy_name = 0, cluster = 0, ra = 0, dec = 0,
                z = 0, z_err = 0, distance = 0, distance_error = 0, lumB = 0, m = 0):
        """Galaxy catalog class... 
        Parameters
        """
        self.index = index
        self.pgc_number = pgc_number
        self.galaxy_name = galaxy_name
        self.cluster = cluster
        self.ra = ra
        self.dec = dec
        self.z = z
        self.z_err = z_err
        self.distance = distance
        self.distance_error = distance_error
        self.lumB = lumB
        self.m = m
        
    def load_astropy_row_glade(self, index, row):
        self.index = index
        self.pgc_number = row['PGC']
        self.galaxy_name = row['Galaxy Name']
        self.cluster = row['Cluster']
        self.ra = row['RA']*np.pi/180.
        self.dec = row['Dec']*np.pi/180.
        self.z = row['z']
        self.distance = row['Distance']
        self.distance_error = row['Distance Error']
        self.m = row['Bmag']
        self.lumB = row['lumB']

    def load_astropy_row_mdc(self, index, row, version):
        self.index = index
        self.ra = row['RA']
        self.dec = row['Dec']
        self.z = row['z']
        if version != "1.0":
            self.m = row['m']
            
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

class galaxyCatalog(object):
    ''' Class for galaxy catalog objects
    '''
    def __init__(self, catalog_file = "",
                catalog_format = 'ascii',
                indexes = 0, 
                dictionary = {}):
        """Galaxy catalog class... 
        Parameters
        """
        self.catalog_file = catalog_file
        self.catalog_format = catalog_format
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
        t = Table.read(self.catalog_file,format=self.catalog_format)
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

        t = Table.read(self.catalog_file,format=self.catalog_format)
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
            kde_m = gaussian_kde(m)
            m_array = np.linspace(15,25,4000)
            m_kde = kde_m.evaluate(m_array)
            mth=m_array[np.where(m_kde==max(m_kde))]
        return mth
