"""
LALinference posterior samples class and methods
Ignacio Magana, Ankan Sur
"""
import numpy as np
import pkg_resources
import healpy as hp
from scipy.stats import gaussian_kde
from scipy import integrate, interpolate, random
from astropy import units as u
from astropy import constants as const
from astropy.table import Table
import h5py

# Global 
posterior_data_path = pkg_resources.resource_filename('gwcosmo', 'data/posterior_samples')
#z_err_fraction = 0.06
#a_err_fraction = 0.08

class posterior_samples(object):
    ''' Class for lalinference posterior samples
    '''
    def __init__(self, lalinference_path = "",lalinference_data=1,distance=1,
                longitude=1,latitude=1,weight=1,nsamples=1,ngalaxies=1):
        """Posterior samples class... (empty by default)
        Parameters
        """
        self.lalinference_path = lalinference_path
        self.lalinference_data = lalinference_data
        self.distance = distance
        self.longitude = longitude
        self.latitude = latitude
        self.weight = weight
        self.nsamples = nsamples

    def load_posterior_samples(self, lalinference_path=posterior_data_path + "/posterior_samples_RR0.dat"):
        """ Loads GW170817 posterior samples by default into class 
            unless lalinference_path points to a samples files. 
            Currently it only supports .dat posterior samples format.
            It also returns distance, ra and dec ...
        """
        self.lalinference_path = lalinference_path
        self.lalinference_data = np.genfromtxt(lalinference_path, names=True)
        self.distance = self.lalinference_data['distance']
        self.longitude = self.lalinference_data['ra']
        self.latitude = self.lalinference_data['dec']
        self.weight = np.ones(len(self.latitude))/(self.distance**2 * np.cos(self.latitude))
        self.nsamples = len(self.weight)

        return self.distance, self.longitude, self.latitude

    def load_posterior_samples_hdf5(self, samples_file_path):
        """ Loads hdf5 posterior samples
        """
        group_name = 'lalinference_mcmc'
        dataset_name = 'posterior_samples'
        f1 = h5py.File(samples_file_path, 'r')
        group = f1[group_name]
        post = group[dataset_name]
        self.distance = post['dist']
        self.longitude = post['ra']
        self.latitude = post['dec']
        self.weight = np.ones(len(self.latitude))/(self.distance**2 * np.cos(self.latitude))
        self.nsamples = len(self.weight)

        return self.distance, self.longitude, self.latitude

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

    def compute_3d_kde(self,coverh_x):
        "Computes 3d KDE"
        three_d_arr = np.vstack((self.longitude, self.latitude, self.distance/coverh_x))
        radecdist = gaussian_kde(three_d_arr)
        return radecdist

    def compute_3d_probability(self, ra, dec, dist, z, lumB, coverh_x, distmax):
        distmin = 0.1 
        
        ngalaxies = len(self.distance) - 1000
        z_err_fraction = 0.06
        a_err_fraction = 0.08

        kde = self.compute_3d_kde(coverh_x)
        pdfnorm = kde.integrate_box(np.asarray([0, -np.pi / 2, 0]), np.asarray([2.0 * np.pi, np.pi / 2, 1.0]))
        
        t = Table([ra,dec,dist,lumB,z],names=('RA','Dec', 'Distance', 'lumB', 'z'))
        nt = t[(np.where((t['Distance'] > distmin) & (t['Distance'] < distmax)))]
        nt = nt[(np.where((nt['RA'] > np.min(self.longitude) - 1.0) \
                          & (nt['RA'] < np.max(self.longitude ) +1.0)))]
        nt = nt[(np.where((nt['Dec'] > np.min(self.latitude) - 1.0) \
                          & (nt['Dec'] < np.max(self.latitude ) +1.0)))]
        
        ra = nt['RA']
        dec = nt['Dec']
        z = nt['z']
        lumB = nt['lumB']

        tmpra = np.transpose(np.tile(ra, (len(self.longitude[ngalaxies:]), 1))) - np.tile(self.longitude[ngalaxies:], (len(ra), 1))
        tmpdec = np.transpose(np.tile(dec, (len(self.latitude[ngalaxies:]), 1))) - np.tile(self.latitude[ngalaxies:], (len(dec), 1))
        tmpm = np.power(tmpra, 2.) + np.power(tmpdec, 2.)
        mask1 = np.ma.masked_where(tmpm > (a_err_fraction**2), tmpm).filled(0)
        mask1 = np.max((mask1 > 0), 1)

        ra = ra[mask1]
        dec = dec[mask1]
        z = z[mask1]
        lumB = lumB[mask1]
        
        tmppdf = kde(np.vstack((ra, dec, z))) / pdfnorm

        return np.sum(tmppdf*lumB/(np.cos(dec)*z**2))