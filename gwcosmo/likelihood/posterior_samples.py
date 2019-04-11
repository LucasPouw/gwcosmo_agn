"""
LALinference posterior samples class and methods
Ignacio Magana, Ankan Sur
"""
import numpy as np
import healpy as hp
from scipy.stats import gaussian_kde
from scipy import integrate, interpolate, random
from astropy import units as u
from astropy import constants as const
from astropy.table import Table
import h5py

class posterior_samples(object):
    ''' Class for lalinference posterior samples
    '''
    def __init__(self,distance=1,longitude=1,latitude=1,weight=1,nsamples=1,ngalaxies=1):
        """Posterior samples class... (empty by default)
        Parameters
        """
        self.distance = distance
        self.longitude = longitude
        self.latitude = latitude
        self.weight = weight
        self.nsamples = nsamples

    def load_posterior_samples(self, samples_file_path):
        """ Loads posterior samples into class 
            Currently it supports .dat, .hdf5, .hdf posterior samples format.
        """
        if samples_file_path[-3:] == 'dat':
            lalinference_data = np.genfromtxt(samples_file_path, names=True)
            self.distance = lalinference_data['distance']
            self.longitude = lalinference_data['ra']
            self.latitude = lalinference_data['dec']
            self.weight = np.ones(len(self.latitude))/(self.distance**2 * np.cos(self.latitude))
            self.nsamples = len(self.weight)

        if samples_file_path[-4:] == 'hdf5':
            group_name = 'lalinference_mcmc'
            dataset_name = 'posterior_samples'
            f1 = h5py.File(samples_file_path, 'r')
            group = f1[group_name]
            lalinference_data = group[dataset_name]
            self.distance = lalinference_data['dist']
            self.longitude = lalinference_data['ra']
            self.latitude = lalinference_data['dec']
            self.weight = np.ones(len(self.latitude))/(self.distance**2 * np.cos(self.latitude))
            self.nsamples = len(self.weight)
            f1.close()

        if samples_file_path[-3:] == 'hdf':
            fp = h5py.File(samples_file_path, 'r')
            self.distance = fp['samples/distance'][:]
            self.longitude = fp['samples/ra'][:]
            self.latitude = fp['samples/dec'][:]
            self.weight = np.ones(len(self.latitude))/(self.distance**2 * np.cos(self.latitude))
            self.nsamples = len(self.weight)
            fp.close()

    def lineofsight_distance(self):
        """
        Takes distance and makes 1-d kde out of it
        """
        return gaussian_kde(self.distance)

    def compute_2d_kde(self):
        two_d_arr = np.vstack((self.longitude, self.latitude))
        radec = gaussian_kde(two_d_arr)
        return radec