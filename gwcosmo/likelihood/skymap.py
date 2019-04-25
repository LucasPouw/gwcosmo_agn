"""
Module to compute and handle skymaps
Rachel Gray, Ignacio Magana, Archisman Ghosh, Ankan Sur
"""
import numpy as np
import scipy.stats
from astropy.io import fits
import healpy as hp
from scipy import interpolate
from scipy.stats import gaussian_kde, norm
from scipy.interpolate import interp1d, UnivariateSpline
import sys

# RA and dec from HEALPix index
def ra_dec_from_ipix(nside, ipix, nest=False):
    (theta, phi) = hp.pix2ang(nside, ipix, nest=nest)
    return (phi, np.pi/2.-theta)

# HEALPix index from RA and dec
def ipix_from_ra_dec(nside, ra, dec, nest=False):
    (theta, phi) = (np.pi/2.-dec, ra)
    return hp.ang2pix(nside, theta, phi, nest=nest)

class skymap2d(object):
    """
    Class for reading in 2d skymaps and returning various things (ie probabilities as specific sky-locations)
    """
    # TODO: tidy this up wrt the functions currently defined outside it.
    def __init__(self,skymap_file):
        self.prob = hp.read_map(skymap_file,nest=True)
        self.npix = len(self.prob)
        self.nside = hp.npix2nside(self.npix)
        self.nested = True
        
    def skyprob(self,ra,dec):
        """
        Takes a value (or array of values) for ra and dec, and computes the probability of the skymap at the location in the sky
        """
        # TODO: change this to interpolate between pixels
        ipix_gal = ipix_from_ra_dec(self.nside,ra,dec,nest=self.nested)
        return self.prob[ipix_gal]

class skymap(object):
    """
    Read a FITS file and return interpolation kernels on the sky.
    TODO: Rework to use ligo.skymap
    """
    def __init__(self, filename):
        """
        Input parameters:
        - filename : FITS file to load from
        """
        prob, header = hp.read_map(filename,field=[0,1,2,3],h=True)
        skymap = prob[0]
        self.nested = True
        self.prob = prob[0]
        self.distmu = prob[1]
        self.distsigma = prob[2]
        self.distnorm = prob[3]
        self.npix = len(self.prob)
        self.nside = hp.npix2nside(self.npix)
        
    def skyprob(self,ra,dec):
        """
        Takes a value (or array of values) for ra and dec, and computes the probability of the skymap at the location in the sky
        """
        # TODO: change this to interpolate between pixels
        ipix_gal = ipix_from_ra_dec(self.nside,ra,dec,nest=self.nested)
        return self.prob[ipix_gal]
    
    def distprob(self):
        return self.distmu[self.prob>1e-5]
    
    def distprob_gaussian(self):
        distmean = np.mean(self.distmu[self.prob>1e-5])
        diststd = np.std(self.distmu[self.prob>1e-5])
        ds = np.linspace(1.,1500.,3000)
        dist_like = norm(distmean,diststd).pdf(ds)/ds**2
        dist_like /= np.trapz(dist_like,ds)
        dist_like_interp = interp1d(ds,dist_like,fill_value=1e-7,bounds_error=False)
        return dist_like_interp

        
    # TODO: rework later
    #def probability(self, ra, dec, dist):
    #    """
    #    returns probability density at given ra, dec, dist
    #    p(ra,dec) * p(dist | ra,dec )
    #    RA, dec : radians
    #    dist : Mpc
    #    """
    #    theta = np.pi/2.0 - dec
    #    # Step 1: find 4 nearest pixels
    #    (pixnums, weights) = \
    #        hp.get_interp_weights(self.nside, theta, ra, nest=self.nested, lonlat = False)
    #    
    #    dist_pdfs = [scipy.stats.norm(loc = self.mean[i], scale = self.sigma[i]) for i in pixnums]
    #    # Step 2: compute p(ra,dec)
    #    # p(ra, dec) = sum_i weight_i p(pixel_i)
    #    probvals = np.array([self.distnorm[i] * dist_pdfs[i].pdf(dist) for i,pixel in enumerate(pixnums)])
    #    skyprob = self.prob[pixnums]      
    #    p_ra_dec = np.sum( weights * probvals * skyprob )
    #    
    #    return(p_ra_dec)
    
    #def skyprob(self,ra,dec):
    #    """
    #    Return the probability of a given sky location
    #    ra, dec: radians
    #    """
    #    ipix_gal = hp.ang2pix(self.nside, np.pi/2.0-dec, ra, nest=self.nested)
    #    return self.prob[ipix_gal]
    
    #def __call__(self, ra, dec, dist):
    #    return self.probability(ra, dec, dist)
