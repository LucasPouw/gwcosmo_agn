"""
Priors
Ignacio Magana, Rachel Gray
"""
from __future__ import absolute_import

import numpy as np
import sys

from scipy.integrate import quad, dblquad
from scipy.stats import ncx2, norm
from scipy.interpolate import splev, splrep
from astropy import constants as const
from astropy import units as u
from bilby.core.prior import Uniform, PowerLaw, PriorDict, Constraint, DeltaFunction
from bilby import gw

import gwcosmo


def constrain_m1m2(parameters):
    converted_parameters = parameters.copy()
    converted_parameters['m1m2'] = parameters['mass_1'] - parameters['mass_2']
    return converted_parameters


def pH0(H0, prior='log'):
    """
    Returns p(H0)
    The prior probability of H0

    Parameters
    ----------
    H0 : float or array_like
        Hubble constant value(s) in kms-1Mpc-1
    prior : str, optional
        The choice of prior (default='log')
        if 'log' uses uniform in log prior
        if 'uniform' uses uniform prior

    Returns
    -------
    float or array_like
        p(H0)
    """
    if prior == 'uniform':
        return np.ones(len(H0))
    if prior == 'log':
        return 1./H0


class mass_distribution(object):
    def __init__(self, name, alpha=1.6, mmin=5, mmax=50, m1=50, m2=50):
        self.name = name
        self.alpha = alpha
        self.mmin = mmin
        self.mmax = mmax
        self.m1 = m1
        self.m2 = m2

        dist = {}

        if self.name == 'BBH-powerlaw':
            dist['mass_1'] = lambda m1s: (np.power(m1s,-self.alpha)/(1-self.alpha))*(np.power(self.mmax,1-self.alpha)-np.power(self.mmin,1-self.alpha))
            dist['mass_2'] = lambda m1s: 1/(m1s-self.mmin)

        if self.name == 'BNS':
            # We assume p(m1,m2)=p(m1)p(m2)
            dist['mass_1'] = lambda m1s: 1/(3-1)
            dist['mass_2'] = lambda m2s: 1/(3-1)

        if self.name == 'NSBH':
            dist['mass_1'] = lambda m1s: (np.power(m1s,-self.alpha)/(1-self.alpha))*(np.power(self.mmax,1-self.alpha)-np.power(self.mmin,1-self.alpha))
            dist['mass_2'] = lambda m2s: 1/(3-1)

        if self.name == 'BBH-constant':
            dist['mass_1'] = DeltaFunction(self.m1)
            dist['mass_2'] = DeltaFunction(self.m2)

        self.dist = dist

    def joint_prob(self, ms1, ms2):

        if self.name == 'BBH-powerlaw':
            # ms1 is not a bug in mass_2. That depends only on that var
            arr_result = self.dist['mass_1'](ms1)*self.dist['mass_2'](ms1)

        if self.name == 'BNS':
            # We assume p(m1,m2)=p(m1)p(m2)
            arr_result = self.dist['mass_1'](ms1)*self.dist['mass_2'](ms2)

        if self.name == 'NSBH':
            arr_result = self.dist['mass_1'](ms1)*self.dist['mass_2'](ms2)

        if self.name == 'BBH-constant':
            arr_result = self.dist['mass_1'].prob(ms1)*self.dist['mass_2'].prob(ms2)

        return arr_result


class distance_distribution(object):
    def __init__(self, name):
        self.name = name

        if self.name == 'BBH-powerlaw':
            dist = PriorDict(conversion_function=constrain_m1m2)
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=15000)

        if self.name == 'BNS':
            dist = PriorDict(conversion_function=constrain_m1m2)
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=1000)

        if self.name == 'NSBH':
            dist = PriorDict(conversion_function=constrain_m1m2)
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=1000)

        if self.name == 'BBH-constant':
            dist = PriorDict()
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=15000)

        self.dist = dist

    def sample(self, N_samples):
        samples = self.dist.sample(N_samples)
        return samples['luminosity_distance']

    def prob(self, samples):
        return self.dist['luminosity_distance'].prob(samples)
