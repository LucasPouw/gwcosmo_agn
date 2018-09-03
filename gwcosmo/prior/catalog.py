"""Module containing functionality for creation and management of galaxy catalogs."""

import numpy as np
from astropy.table import Table
import pkg_resources

# Global
catalog_data_path = pkg_resources.resource_filename('gwcosmo', 'data/catalog_data/')
groupmembers4993 = np.genfromtxt(catalog_data_path + "NGC4993group.txt", usecols=0) #NGC 4993's group
pgc_KTgroups,groupgal = np.genfromtxt(catalog_data_path + "KTgroups.txt", usecols=(0,4), unpack=True)

class galaxy(object):
    ''' Class for galaxy objects
    '''
    def __init__(self,
                index = 0,
                astropy_row = 0,
                pgc_number = 0,
                galaxy_name = 0,
                cluster = 0,
                ra = 0,
                dec = 0,
                z = 0,
                z_err = 0,
                distance = 0,
                distance_error = 0,
                angle_error = 0,
                abs_mag_r = 0,
                abs_mag_k = 0,
                lumB = 0,
                lumK=0):
        """Galaxy catalog class... 
        Parameters
        """
        self.index = index
        self.astropy_row = astropy_row    
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
        
    def load_astropy_row(self, index, row):
        self.index = index
        self.astropy_row = row
        self.pgc_number = row['PGC']
        self.galaxy_name = row['Galaxy Name']
        self.cluster = row['Cluster']
        self.ra = row['RA']
        self.dec = row['Dec']
        self.z = row['z']
        self.distance = row['Distance']
        self.distance_error = row['Distance Error']
        self.abs_mag_r = row['abs_mag_r']
        self.abs_mag_k = row['abs_mag_k']

class galaxyCatalog(object):
    ''' Class for galaxy catalog objects
    '''
    def __init__(self, catalog_file = catalog_data_path + "gladecatalogv2.3.dat",
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

    def load_glade_catalog(self):
        t = Table.read(self.catalog_file,format=self.catalog_format)
        galaxies={}
        nGal = len(t)
        for k in range(0,nGal):
            gal = galaxy()
            gal.load_astropy_row(k,t[k])
            galaxies[str(k)]= gal
        self.dictionary = galaxies
        self.indexes = np.arange(nGal)

    def get_galaxy(self,index):
        return self.dictionary[str(int(index))]