"""
LALinference posterior samples class and methods
"""
__author__ = "Ignacio Magana Hernandez <ignacio.magana@ligo.org>"


"""Module containing functionality for creation and management of completion functions."""
import numpy as np
import pkg_resources
from scipy.stats import gaussian_kde
from scipy import integrate, interpolate, random

# Global 
posterior_data_path = pkg_resources.resource_filename('gwcosmo', 'data/posterior_samples')
class posterior_samples(object):
    ''' Class for lalinference posterior samples
    '''
    def __init__(self, lalinference_path = posterior_data_path + "/posterior_samples_RR0.dat",
    			lalinference_data=1,distance=1,longitude=1,latitude=1,weight=1,nsamples=1,ngalaxies=1):
        """posterior samples class... 
        Parameters
        """
        self.lalinference_path = lalinference_path
        self.lalinference_data = lalinference_data
        self.distance = distance
        self.longitude = longitude
        self.latitude = latitude
        self.weight = weight
        self.nsamples = nsamples
        self.ngalaxies = ngalaxies

    def load_posterior_samples(self):
        lalinference_data = np.genfromtxt(self.lalinference_path, names=True)
    	distance = lalinference_data['distance']
    	longitude = lalinference_data['ra']
    	latitude = lalinference_data['dec']
    	weight = np.ones(len(latitude))/(distance*distance*np.cos(latitude))
    	nsamples = len(weight)

        self.lalinference_data = lalinference_data
        self.distance = distance
        self.longitude = longitude
        self.latitude = latitude
        self.weights = weights
        self.nsamples = nsamples
        self.ngalaxies = ngalaxies

        return lalinference_data,distance,longitude,latitude,weights,ngalaxies

    def lineofsight_distance(self, distance):
        """
        Takes distance and makes 1-d kde out of it
        """
        return gaussian_kde(self.distance)

    def dist_prior_corr(self, distance):
        """
        Change of prior from uniform in volume to uniform in distance
        """
        xx = np.linspace(0.9*np.min(self.distance), 1.1*np.max(self.distance), 100.)
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

 




