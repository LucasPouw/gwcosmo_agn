"""
LALinference posterior samples class and methods
"""
__author__ = "Ignacio Magana Hernandez <ignacio.magana@ligo.org>"


"""Module containing functionality for creation and management of completion functions."""
import numpy as np
import pkg_resources
import healpy as hp
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


    def ra_dec_from_ipix(self, nside, ipix):
        (theta, phi) = hp.pix2ang(nside, ipix)
        return (phi, np.pi/2.-theta)

    # HEALPix index from RA and dec
    def ipix_from_ra_dec(self, nside, ra, dec):
        (theta, phi) = (np.pi/2.-dec, ra)
        return hp.ang2pix(nside, theta, phi)


    def compute_2d_sky_prob(self, sky_level, nside, ra, dec):
   
        skymap = hp.read_map(skymap_file)
        nside = hp.npix2nside(len(skymap))
        ipix_gal = ipix_from_ra_dec(nside, ra, dec)

        # Height of probability contour corresponding to confidence level set above
        skyconf_obj = confidence(skymap)
        sky_height = skyconf_obj.height_from_level(sky_level)   

        # Pixels of skymap inside the probability contour
        ipix_above_height, = np.where(skymap > sky_height)

        # Indices of galaxies inside the probability contour
        idx_gal_above_height = np.array([ig in ipix_above_height for ig in ipix_gal]) 

       
    def compute_3d_kde(self, catalog, distmin, distmax):
        "Computes 3d KDE"
        catalog = gwcosmo.catalog.galaxyCatalog()
        catalog.load_glade_catalog()

        #some_galaxy = catalog.get_galaxy(1000)

        ra = some_galaxy.ra
        dec = some_galaxy.dec

        nt = t[np.argmax(t['Distance'] > distmin):np.argmin(t['Distance'] < distmax)]

        nt.sort('RA')
        nt = nt[np.argmax(nt['RA'] > np.min(self.longitude) - 1.0):np.argmin(nt['RA'] < np.max(self.longitude ) +1.0)]

        nt.sort('Dec')
        nt = nt[np.argmax(nt['Dec'] > np.min(self.latitude) - 1.0):np.argmin(nt['Dec'] < np.max(self.latitude ) + 1.0)]

        catalog.catalog = nt
        (pgcCat, ra, dec, dist, z, lumB, angle_error, dist_err, z_err) = catalog.extract_galaxies_table()

        tmpra = np.transpose(np.tile(ra, (len(longitude[self.ngalaxies:]), 1))) - np.tile(longitude[self.ngalaxies:], (len(ra), 1))
        tmpdec = np.transpose(np.tile(dec, (len(latitude[self.ngalaxies:]), 1))) - np.tile(latitude[self.ngalaxies:], (len(dec), 1))
        tmpm = np.power(tmpra, 2.) + np.power(tmpdec, 2.)
        mask1 = np.ma.masked_where(tmpm > (a_err_fraction**2), tmpm).filled(0)
        mask1 = np.max((mask1 > 0), 1)

        ra = ra[mask1]
        dec = dec[mask1]
        dist = dist[mask1]
        z = z[mask1]
        lumB = lumB[mask1]
        print "No. of used galaxies %i" % (len(ra))

        # Calculate posterior
        for k, x in enumerate(hzero):
            coverh = (const.c.to('km/s') / (x * u.km / u.s / u.Mpc)).value
            pdf = stats.gaussian_kde(np.vstack((longitude[nsamples:], latitude[nsamples:], distance[nsamples:] / coverh)))
            pdfnorm = pdf.integrate_box(np.asarray([0, -np.pi / 2, 0]), np.asarray([2.0 * np.pi, np.pi / 2, 1.0]))
            tmppdf = pdf(np.vstack((ra, dec, z))) / pdfnorm
            ph[k] = np.sum(tmppdf * lumB / (np.cos(dec) * z**2))
            completion = cfactor * completionFun.generateCompletion(lumB, dist, distance / coverh, useNGC4993only) / (4.0 * np.pi)
            epsilon = 0.5 * (1 - np.tanh(3.3 * np.log(distance / 80.)))
            ph[k] = (ph[k] + np.mean((completion ) / ((distance / coverh)**2))) 




