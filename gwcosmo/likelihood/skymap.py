import numpy as np
import scipy.stats
from astropy.io import fits
import healpy as hp

class FITSkernel(object):
    """
    Read a FITS file and return interpolation kernels on the sky.
    TODO: Rework to use ligo.skymap
    """
    def __init__(self, filename):
        """
        Input parameters:
        - filename : FITS file to load from
        """
        hdulist = fits.open(filename)
        hdu = hdulist[1]
        tbdata = hdu.data
        self.nested = True
        self.prob = tbdata['PROB']
        self.mean = tbdata['DISTMU']
        self.sigma = tbdata['DISTSIGMA']
        self.distnorm = tbdata['DISTNORM']
        self.npix = len(self.prob)
        self.nside = hp.npix2nside(self.npix)
        colat, self.ra = hp.pix2ang(self.nside, range(len(self.prob)), nest=self.nested)
        self.dec = np.pi/2.0 - colat
        
    def probability(self, ra, dec, dist):
        """
        returns probability density at given ra, dec, dist
        p(ra,dec) * p(dist | ra,dec )
        RA, dec : radians
        dist : Mpc
        """
        theta = np.pi/2.0 - dec
        # Step 1: find 4 nearest pixels
        (pixnums, weights) = \
            hp.get_interp_weights(self.nside, theta, ra, nest=self.nested, lonlat = False)
        
        dist_pdfs = [scipy.stats.norm(loc = self.mean[i], scale = self.sigma[i]) for i in pixnums]
        # Step 2: compute p(ra,dec)
        # p(ra, dec) = sum_i weight_i p(pixel_i)
        probvals = np.array([self.distnorm[i] * dist_pdfs[i].pdf(dist) for i,pixel in enumerate(pixnums)])
        skyprob = self.prob[pixnums]      
        p_ra_dec = np.sum( weights * probvals * skyprob )
        
        return(p_ra_dec)

    def skyprob(self,ra,dec):
        """
        Return the probability of a given sky location
        ra, dec: radians
        """
        ipix_gal = hp.ang2pix(self.nside, np.pi/2.0-dec, ra, nest=self.nested)
        return self.prob[ipix_gal]
    
    def __call__(self, ra, dec, dist):
        return self.probability(ra, dec, dist)
