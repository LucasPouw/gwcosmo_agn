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

    def load_posterior_samples(self,event):
        """ Loads GW170817 posterior samples by default into class 
            Currently it only supports .dat posterior samples format.
        """
        lalinference_data = np.genfromtxt(event, names=True)
        self.distance = lalinference_data['distance']
        self.longitude = lalinference_data['ra']
        self.latitude = lalinference_data['dec']
        self.weight = np.ones(len(self.latitude))/(self.distance**2 * np.cos(self.latitude))
        self.nsamples = len(self.weight)

    def load_posterior_samples_hdf5(self, samples_file_path):
        """ Loads hdf5 posterior samples
        """
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

    def load_posterior_samples_hdf(self, samples_file_path):
        """ Loads hdf posterior samples
        """
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

    def dist_prior_corr(self):
        """
        Change of prior from uniform in volume to uniform in distance
        """
        # TODO: decide on resolution of grid which won't cause issues down the line
        dist_kde = self.lineofsight_distance()
        xx = np.linspace(0.9*np.min(self.distance), 1.1*np.max(self.distance), 1000.)
        yy = dist_kde(xx)/xx**2. 
        yy /= np.sum(yy)*(xx[1]-xx[0])
        # Interpolation of normalized prior-corrected distribution
        try:
            # The following works only on recent python versions
            dist_support = interpolate.InterpolatedUnivariateSpline(xx, yy, ext=1)
        except TypeError:
            # A workaround to prevent bounds error in earlier python versions
            dist_interp = interpolate.InterpolatedUnivariateSpline(xx, yy)
            def dist_support(x):
                if (x>=xx[0]) and (x<=xx[-1]):
                    return dist_interp(x)
                return 0.
        dist_support = np.vectorize(dist_support)
        return dist_support

    def compute_2d_kde(self):
        two_d_arr = np.vstack((self.longitude, self.latitude))
        radec = gaussian_kde(two_d_arr)
        return radec    
        
    def sky_prior_corr(self):
        """
        Remove uniform in sky prior
        """
        # Currently not working
        # TODO: make this work in similar way to dist_prior_corr
        # Also note, might have issues using this method due to wrapping around on the sky.  Move to skymaps?
        sky_kde = self.compute_2d_kde()
        ww = np.linspace(0.9*np.min(self.longitude), 1.1*np.max(self.longitude), 100.)
        xx = np.linspace(0.9*np.min(self.latitude), 1.1*np.max(self.latitude), 100.)
        yy = sky_kde([ww,xx])*4.0*np.pi/np.cos(xx)
        yy /= np.sum(yy)*(xx[1]-xx[0])
        # Interpolation of normalized prior-corrected distribution
        try:
            # The following works only on recent python versions
            sky_support = interpolate.InterpolatedUnivariateSpline(xx, yy, ext=1)
        except TypeError:
            # A workaround to prevent bounds error in earlier python versions
            sky_interp = interpolate.InterpolatedUnivariateSpline(xx, yy)
            def sky_support(x):
                if (x>=xx[0]) and (x<=xx[-1]):
                    return sky_interp(x)
                return 0.
        sky_support = np.vectorize(sky_support)
        return sky_support