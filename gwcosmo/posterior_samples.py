"""
LALinference posterior samples class and methods
"""
__author__ = "Ignacio Magana Hernandez <ignacio.magana@ligo.org>"


"""Module containing functionality for creation and management of completion functions."""
import numpy as np
#import pkg_resources

# Global 
#posterior_data_path = pkg_resources.resource_filename('gwcosmo', 'likelihood/posterior_samples')
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