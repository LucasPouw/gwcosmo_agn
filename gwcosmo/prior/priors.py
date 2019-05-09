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

import gwcosmo


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


def BBH_mass_distribution(N, mmin=5., mmax=40., alpha=-1):
    """
    Returns p(m1,m2)
    The prior on the mass distribution that follows a power law (or flat in
    log when alpha = -1) for BBHs.

    Parameters
    ----------
    N : integer
        Number of masses sampled
    mmin : float
        minimum mass
    mmax : float
        maximum mass
    alpha : float
        slope of the power law

    Returns
    -------
    float or array_like
        m1, m2
    """
    u = np.random.rand(N)
    if alpha != -1:
        m1 = (u*(mmax**(alpha+1)-mmin**(alpha+1)) +
              mmin**(alpha+1))**(1.0/(alpha+1))
    else:
        m1 = np.exp(u*(np.log(mmax)-np.log(mmin))+np.log(mmin))
    m2 = np.random.uniform(low=5.0, high=m1)
    return m1, m2


def BNS_gaussian_distribution(N, mean=1.35, sigma=0.15):
    """
    Returns p(m1,m2)
    The prior on the mass distribution that follows gaussian for BNSs.

    Parameters
    ----------
    N : integer
        Number of masses sampled
    mean : float
        mean of gaussian dist
    sigma : float
        std of gaussian dist

    Returns
    -------
    float or array_like
        mass1, mass2
    """
    mass1 = []
    mass2 = []
    while len(mass1) < N:
        m1 = np.random.normal(mean, sigma)
        m2 = np.random.normal(mean, sigma)
        if m2 > m1:
            m3 = m2
            m2 = m1
            m1 = m3
        mass1.append(m1)
        mass2.append(m2)
    mass1 = np.array(mass1)
    mass2 = np.array(mass2)
    return mass1, mass2


def BNS_uniform_distribution(N, mmin=1., mmax=3.):
    """
    Returns p(m1,m2)
    The prior on the mass distribution that follows gaussian for BNSs.

    Parameters
    ----------
    N : integer
        Number of masses sampled
    mmin : float
        minimum mass
    mmax : float
        maximum mass

    Returns
    -------
    float or array_like
        mass1, mass2
    """
    mass1 = []
    mass2 = []
    while len(mass1) < N:
        m1 = np.random.uniform(mmin, mmax)
        m2 = np.random.uniform(mmin, mmax)
        if m2 > m1:
            m3 = m2
            m2 = m1
            m1 = m3
        mass1.append(m1)
        mass2.append(m2)
    mass1 = np.array(mass1)
    mass2 = np.array(mass2)
    return mass1, mass2
