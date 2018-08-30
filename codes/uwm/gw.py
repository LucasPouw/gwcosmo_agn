"""Constructs GW likelihood."""

"""Module containing functionality for creation and management of completion functions."""
import numpy as np
import pkg_resources

# Global 
posterior_data_path = pkg_resources.resource_filename('gwcosmo', 'likelihood/posterior_samples')
ns = 5042
nsg = 20

class posterior_samples(object):
    ''' Class for GW likelihood objects
    '''
    def __init__(self, lalinference_path= posterior_data_path + "/posterior_samples_RR0.dat",
    			lalinference_data=1,distance=1,longitude=1,latitude=1,weight=1,nsamples=1,ngalaxies=1):
        """GW likelihood class... 
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
    	nsamples = len(weight)-(ns)
    	ngalaxies = len(weight)-(nsg)

        self.lalinference_data = lalinference_data
        self.distance = distance
        self.longitude = longitude
        self.latitude = latitude
        self.weight = weight
        self.nsamples = nsamples
        self.ngalaxies = ngalaxies

        return lalinference_data,distance,longitude,latitude,weight,nsamples,ngalaxies