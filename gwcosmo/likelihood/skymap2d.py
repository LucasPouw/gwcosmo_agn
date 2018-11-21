"""
Module to compute 2d sky probability
Archisman Ghosh, Ankan Sur, Rachel Gray
"""
import numpy as np, healpy as hp
from scipy import interpolate
import sys

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


def ipix_from_ra_dec(nside, ra, dec, nest=False):
    (theta, phi) = (np.pi/2.-dec, ra)
    return hp.ang2pix(nside, theta, phi, nest=nest)
