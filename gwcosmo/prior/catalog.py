"""Module containing functionality for creation and management of galaxy catalogs.
Ignacio Magana
"""

import numpy as np
from astropy.table import Table
import pkg_resources

# Global
catalog_data_path = pkg_resources.resource_filename('gwcosmo', 'data/catalog_data/')

class galaxy(object):
    ''' Class for galaxy objects
    '''
    def __init__(self, index = 0, pgc_number = 0, galaxy_name = 0, cluster = 0, ra = 0, dec = 0,
                z = 0, z_err = 0, distance = 0, distance_error = 0, angle_error = 0, 
                abs_mag_r = 0, abs_mag_k = 0, lumB = 0, lumK=0, m = 0):
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
        self.angle_error = angle_error
        self.abs_mag_r = abs_mag_r
        self.abs_mag_k = abs_mag_k
        self.lumB = lumB
        self.lumK = lumK
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
        self.abs_mag_r = row['abs_mag_r']
        self.abs_mag_k = row['abs_mag_k']

    def load_astropy_row_mdc(self, index, row, version):
        self.index = index
        self.ra = row['RA']
        self.dec = row['Dec']
        self.z = row['z']
        if version != "1.0":
            self.m = row['m']

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

    def load_glade_catalog(self, version='maya'):
        if version == 'corrected':
            self.catalog_file = catalog_data_path + "gladecatalogv2.3_corrected.dat"
            t = Table.read(self.catalog_file,format=self.catalog_format)
        if version == 'original':
            self.catalog_file = catalog_data_path + "gladecatalogv2.3.dat"
            t = Table.read(self.catalog_file,format=self.catalog_format)
        if version == 'maya': #Here for testing purposes
            self.catalog_file = catalog_data_path + "glade23_maya_cuts.txt"
            pgcsel,rasel,decsel,zsel_group,bmagsel,bMagsel,kmagsel,kMagsel = np.genfromtxt(self.catalog_file,unpack=True)
            __ = np.ones(len(pgcsel))
            t = Table([pgcsel, __, __, rasel, decsel, zsel_group, __, __, bMagsel, kMagsel ],
                names=['PGC','Galaxy Name','Cluster','RA', 'Dec', 'z', 'Distance','Distance Error','abs_mag_r','abs_mag_k'])

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
        
    def nGal(self):
        return len(self.dictionary)

    def get_galaxy(self,index):
        return self.dictionary[str(int(index))]

