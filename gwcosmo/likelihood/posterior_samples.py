"""
LALinference posterior samples class and methods
Ignacio Magana, Ankan Sur
"""
import numpy as np
import healpy as hp
from scipy.stats import gaussian_kde
from scipy import integrate, interpolate, random
from astropy import units as u
from astropy import constants as const
from astropy.table import Table
import h5py


class posterior_samples(object):
    """
    Posterior samples class and methods.

    Parameters
    ----------
    posterior_samples : Path to posterior samples file to be loaded.
    """
    def __init__(self, posterior_samples=None):
        self.posterior_samples = posterior_samples
        try:
            self.load_posterior_samples()
        except:
            print("No posterior samples were specified")

    def load_posterior_samples(self):
        """
        Method to handle different types of posterior samples file formats.
        Currently it supports .dat (LALinference), .hdf5 (GWTC-1), 
        .h5 (PESummary) and .hdf (pycbcinference) formats.
        """
        if self.posterior_samples[-3:] == 'dat':
            samples = np.genfromtxt(self.posterior_samples, names=True)
            try:
                self.distance = samples['dist']
            except KeyError:
                try:
                    self.distance = samples['distance']
                except KeyError:
                    print("No distance samples found.")
            self.ra = samples['ra']
            self.dec = samples['dec']
            self.mass_1 = samples['mass_1']
            self.mass_2 = samples['mass_2']
            self.nsamples = len(self.distance)

        if self.posterior_samples[-4:] == 'hdf5':
            if self.posterior_samples[-11:] == 'GWTC-1.hdf5':
                if self.posterior_samples[-20:] == 'GW170817_GWTC-1.hdf5':
                    dataset_name = 'IMRPhenomPv2NRT_lowSpin_posterior'
                else:
                    dataset_name = 'IMRPhenomPv2_posterior'
                file = h5py.File(self.posterior_samples, 'r')
                data = file[dataset_name]
                self.distance = data['luminosity_distance_Mpc']
                self.ra = data['right_ascension']
                self.dec = data['declination']
                self.mass_1 = data['m1_detector_frame_Msun']
                self.mass_2 = data['m2_detector_frame_Msun']
                self.nsamples = len(self.distance)
                file.close()

        if self.posterior_samples[-2:] == 'h5':
            file = h5py.File(self.posterior_samples, 'r')
            approximants = ['C01:PhenomPNRT-HS', 'C01:NRSur7dq4',
                            'C01:IMRPhenomPv3HM', 'C01:IMRPhenomPv2']
            for approximant in approximants:
                try:
                    data = file[approximant]
                    print("Using "+approximant+" posterior")
                    break
                except KeyError:
                    continue

            self.distance = data['posterior_samples']['luminosity_distance']
            self.ra = data['posterior_samples']['ra']
            self.dec = data['posterior_samples']['dec']
            self.mass_1 = data['posterior_samples']['mass_1']
            self.mass_2 = data['posterior_samples']['mass_2']
            self.nsamples = len(self.distance)
            file.close()

        if self.posterior_samples[-3:] == 'hdf':
            file = h5py.File(self.posterior_samples, 'r')
            self.distance = file['samples/distance'][:]
            self.ra = file['samples/ra'][:]
            self.dec = file['samples/dec'][:]
            self.mass_1 = file['samples/mass_1'][:]
            self.mass_2 = file['samples/mass_2'][:]
            self.nsamples = len(self.distance)
            file.close()

    def marginalized_distance(self):
        """
        Computes the marginalized distance posterior KDE.
        """
        return gaussian_kde(self.distance)

    def marginalized_sky(self):
        """
        Computes the marginalized sky localization posterior KDE.
        """
        return gaussian_kde(np.vstack((self.ra, self.dec)))
