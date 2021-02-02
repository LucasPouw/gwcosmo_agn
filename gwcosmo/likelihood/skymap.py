"""
Module to compute and handle skymaps
Rachel Gray, Ignacio Magana, Archisman Ghosh, Ankan Sur
"""
import numpy as np
import scipy.stats
from astropy.io import fits
import healpy as hp
from scipy import interpolate
from scipy.stats import norm
import sys


# RA and dec from HEALPix index
def ra_dec_from_ipix(nside, ipix, nest=False):
    (theta, phi) = hp.pix2ang(nside, ipix, nest=nest)
    return (phi, np.pi/2.-theta)


# HEALPix index from RA and dec
def ipix_from_ra_dec(nside, ra, dec, nest=False):
    (theta, phi) = (np.pi/2.-dec, ra)
    return hp.ang2pix(nside, theta, phi, nest=nest)


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
        try:
            prob, header = hp.read_map(filename, field=[0, 1, 2, 3],
                                       h=True, nest=True)
            self.prob = prob[0]
            self.distmu = prob[1]
            self.distsigma = prob[2]
            self.distnorm = prob[3]
        except IndexError:
            self.prob = hp.read_map(filename, nest=True)
            self.distmu = np.ones(len(self.prob))
            self.distsigma = np.ones(len(self.prob))
            self.distnorm = np.ones(len(self.prob))

        self.nested = True
        self.npix = len(self.prob)
        self.nside = hp.npix2nside(self.npix)
        colat, self.ra = hp.pix2ang(self.nside, range(len(self.prob)),
                                    nest=self.nested)
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
            hp.get_interp_weights(self.nside, theta, ra,
                                  nest=self.nested, lonlat=False)

        dist_pdfs = [scipy.stats.norm(loc=self.mean[i], scale=self.sigma[i])
                     for i in pixnums]
        # Step 2: compute p(ra,dec)
        # p(ra, dec) = sum_i weight_i p(pixel_i)
        probvals = np.array([self.distnorm[i] * dist_pdfs[i].pdf(dist)
                            for i, pixel in enumerate(pixnums)])
        skyprob = self.prob[pixnums]
        p_ra_dec = np.sum(weights * probvals * skyprob)

        return(p_ra_dec)

    def skyprob(self, ra, dec):
        """
        Return the probability of a given sky location
        ra, dec: radians
        """
        ipix_gal = self.indices(ra,dec)
        return self.prob[ipix_gal]

    def indices(self, ra, dec):
        """
        Return the index of the skymap pixel that contains the coordinate ra,dec
        """
        return hp.ang2pix(self.nside, np.pi/2.0-dec, ra, nest=self.nested)

    def marginalized_distance(self):
        mu = self.distmu[(self.distmu<np.inf) & (self.distmu>0)]
        distmin = 0.5*min(mu)
        distmax = 2*max(mu)
        dl = np.linspace(distmin, distmax, 200)
        dp_dr = [np.sum(self.prob * r**2 * self.distnorm *
                        norm(self.distmu, self.distsigma).pdf(r)) for r in dl]
        return dl, dp_dr

    def lineofsight_distance(self, ra, dec):
        ipix = ipix_from_ra_dec(self.nside, ra, dec, nest=self.nested)
        mu = self.distmu[(self.distmu<np.inf) & (self.distmu>0)]
        distmin = 0.5*min(mu)
        distmax = 2*max(mu)
        r = np.linspace(distmin, distmax, 200)
        dp_dr = r**2 * self.distnorm[ipix] * norm(self.distmu[ipix],
                                                  self.distsigma[ipix]).pdf(r)
        return r, dp_dr

    def probability(self, ra, dec, dist):
        """
        returns probability density at given ra, dec, dist
        p(ra,dec) * p(dist | ra,dec )
        RA, dec : radians
        dist : Mpc
        """
        theta = np.pi/2.0 - dec
        # Step 1: find 4 nearest pixels
        (pixnums, weights) = hp.get_interp_weights(self.nside,
                                                   theta, ra, nest=self.nested,
                                                   lonlat=False)

        dist_pdfs = [norm(loc=self.distmu[i], scale=self.distsigma[i])
                     for i in pixnums]
        # Step 2: compute p(ra,dec)
        # p(ra, dec) = sum_i weight_i p(pixel_i)
        probvals = np.array([self.distnorm[i] * dist_pdfs[i].pdf(dist)
                            for i, pixel in enumerate(pixnums)])
        skyprob = self.prob[pixnums]
        p_ra_dec = np.sum(weights * probvals * skyprob)

        return(p_ra_dec)

    def above_percentile(self, thresh, nside=None):
        """Returns indices of array within the given threshold
        credible region."""
        prob = self.prob
        if nside != None:
            new_prob = hp.pixelfunc.ud_grade(self.prob, nside, order_in='NESTED', order_out='NESTED')
            prob = new_prob/np.sum(new_prob) #renormalise
            
        #  Sort indicies of sky map
        ind_sorted = np.argsort(-prob)
        #  Cumulatively sum the sky map
        cumsum = np.cumsum(prob[ind_sorted])
        #  Find indicies contained within threshold area
        lim_ind = np.where(cumsum > thresh)[0][0]
        return ind_sorted[:lim_ind], prob

    def samples_within_region(self, ra, dec, thresh, nside=None):
        """Returns boolean array of whether galaxies are within
        the sky map's credible region above the given threshold"""
        skymap_ind = self.above_percentile(thresh, nside=nside)[0]
        samples_ind = hp.ang2pix(nside, np.pi/2.0-dec, ra, nest=self.nested)
        return np.in1d(samples_ind, skymap_ind)
        
    def region_with_sample_support(self, ra, dec, thresh, nside=None):
        """
        Finds fraction of sky with catalogue support, and corresponding
        fraction of GW sky probability
        """
        skymap_ind, skymap_prob = self.above_percentile(thresh,nside=nside)
        samples_ind = hp.ang2pix(nside, np.pi/2.0-dec, ra, nest=True)
        ind = np.in1d(skymap_ind, samples_ind)
        fraction_of_sky = np.count_nonzero(ind)/len(skymap_prob)
        GW_prob_in_fraction_of_sky = np.sum(skymap_prob[skymap_ind[ind]])
        return fraction_of_sky,GW_prob_in_fraction_of_sky

    def pixel_split(self, ra, dec, nside):
        """
        Convert a catalogue to a HEALPix map of mth per resolution
        element (by taking the faintest object in each pixel).

        Parameters
        ----------
        ra, dec : (ndarray, ndarray)
            Coordinates of the sources in radians.

        nside : int
            HEALPix nside of the target map

        Return
        ------
        res : ndarray
            arrays of galaxy indices corresponding to each pixel
        idx : array
            healpy index of each pixel containing at least one galaxy
            
        ra[res[i]] returns the ra values of each galaxy in skymap pixel
        idx[i].

        """

        # The number of pixels based on the chosen value of nside
        npix = hp.nside2npix(nside)

        # conver to theta, phi
        theta = np.pi/2.0 - dec
        phi = ra
        
        # convert to HEALPix indices (each galaxy is assigned to a single healpy pixel)
        indices = hp.ang2pix(nside, theta, phi, nest=self.nested)
        
        # sort the indices into ascending order
        idx_sort = np.argsort(indices)
        sorted_indices = indices[idx_sort]
        
        # idx: the healpy index of each pixel containing a galaxy (arranged in ascending order)
        # idx_start: the index of 'sorted_indices' corresponding to each new pixel
        # count: the number of galaxies in each pixel
        idx, idx_start,count = np.unique(sorted_indices,return_counts=True,return_index=True)
        
        # splits indices into arrays - 1 per pixel
        res = np.split(idx_sort, idx_start[1:])

        return res, idx
        
