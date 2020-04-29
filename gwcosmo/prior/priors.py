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

    
def BBH_mass_distribution(N, mmin=5., mmax=40., alpha=1.6):
    """
    Returns p(m1,m2)
    The prior on the mass distribution that follows a power
    law for BBHs.

    Parameters
    ----------
    N : integer
        Number of masses sampled
    mmin : float
        minimum mass
    mmax : float
        maximum mass
    alpha : float
        slope of the power law p(m) = m^-\alpha where alpha > 0

    Returns
    -------
    float or array_like
        m1, m2
    """
    alpha_ = -1*alpha
    u = np.random.rand(N)
    if alpha_ != -1:
        m1 = (u*(mmax**(alpha_+1)-mmin**(alpha_+1)) +
              mmin**(alpha_+1))**(1.0/(alpha_+1))
        print('Powerlaw mass distribution with alpha = ' + str(alpha))
    else:
        m1 = np.exp(u*(np.log(mmax)-np.log(mmin))+np.log(mmin))
        print('Flat in log mass distribution')
    m2 = np.random.uniform(low=mmin, high=m1)
    return m1, m2


def BBH_constant_mass(N, M1=50., M2=50.):
    """
    Returns p(M1,M2) for given masses.

    Parameters
    ----------
    N : integer
        Number of masses sampled
    M1 : float
        source mass
    M2 : float
        source mass
    -------
    float or array_like
        m1, m2
    """
    m1=[]
    m2=[]
    while len(m1) < N:
        m1.append(M1)
        m2.append(M2)
    m1 = np.array(m1)
    m2 = np.array(m2)
    print('BBH source masses M1 = ' + str(M1) + ' M2 = '+ str(M2))
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
