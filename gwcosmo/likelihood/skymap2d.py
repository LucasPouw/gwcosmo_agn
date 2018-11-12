#!/usr/bin/env python

'''
Module to compute 2d sky probability an select galaxies 

authors= Archisman Ghosh, Ankan Sur, Rachel Gray

'''

import numpy as np, healpy as hp
from scipy import interpolate
import sys

class confidence(object):
    def __init__(self, counts):
        # Sort in descending order in frequency
        self.counts_sorted = np.sort(counts.flatten())[::-1]
        # Get a normalized cumulative distribution from the mode
        self.norm_cumsum_counts_sorted = np.cumsum(self.counts_sorted) / np.sum(counts)
        # Set interpolations between heights, bins and levels
        self._set_interp()
    def _set_interp(self):
        self._length = len(self.counts_sorted)
        # height from index
        self._height_from_idx = interpolate.interp1d(np.arange(self._length), self.counts_sorted, bounds_error=False, fill_value=0.)
        # index from height
        self._idx_from_height = interpolate.interp1d(self.counts_sorted[::-1], np.arange(self._length)[::-1], bounds_error=False,               fill_value=self._length)
        # level from index
        self._level_from_idx = interpolate.interp1d(np.arange(self._length), self.norm_cumsum_counts_sorted, bounds_error=False, fill_value=1.)
        # index from level
        self._idx_from_level = interpolate.interp1d(self.norm_cumsum_counts_sorted, np.arange(self._length), bounds_error=False, fill_value=self._length)
    def level_from_height(self, height):
        return self._level_from_idx(self._idx_from_height(height))
    def height_from_level(self, level):
        return self._height_from_idx(self._idx_from_level(level))


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


def twodskyprob(skymap_file, ra, dec, z_gal, z_min, z_max, sky_level=0.99):

    skymap = hp.read_map(skymap_file)
    nside = hp.npix2nside(len(skymap))

    # Map each galaxy to a pixel on the skymap
    ipix_gal = ipix_from_ra_dec(nside, ra, dec)

    # Height of probability contour corresponding to confidence level set above
    skyconf_obj = confidence(skymap)
    sky_height = skyconf_obj.height_from_level(sky_level)

    # Pixels of skymap inside the probability contour
    ipix_above_height, = np.where(skymap > sky_height)

    # Indices of galaxies inside the probability contour
    idx_gal_above_height = np.array([ig in ipix_above_height for ig in ipix_gal])
    
    # Impose a cut on z (min and max values chosen above)
    valid_idx, = np.where((z_min<z_gal)&(z_gal<z_max)&(idx_gal_above_height))
    sys.stderr.write('%d galaxies in %d%% sky.\n'%(len(valid_idx), int(100*sky_level)))
    valid_gal_ra_arr = ra[valid_idx]
    valid_gal_dec_arr = dec[valid_idx]
    valid_gal_z_arr = z_gal[valid_idx]
    valid_gal_sky_prob_arr = skymap[ipix_gal[valid_idx]]
    valid_gal_sky_conf_arr = np.vectorize(skyconf_obj.level_from_height)(valid_gal_sky_prob_arr)

    return valid_gal_ra_arr, valid_gal_dec_arr, valid_gal_z_arr, valid_gal_sky_prob_arr, valid_gal_sky_conf_arr



