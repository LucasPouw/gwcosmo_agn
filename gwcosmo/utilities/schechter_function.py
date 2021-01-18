"""
Module with Schechter magnitude function:
(C) Walter Del Pozzo (2014)
Rachel Gray
"""
from numpy import *
from scipy.integrate import quad
import numpy as np

class SchechterMagFunction(object):
    def __init__(self, Mstar_obs=-19.70, alpha=-1.07, phistar=1.):
        self.Mstar_obs = Mstar_obs
        self.phistar = phistar
        self.alpha = alpha
        self.norm = None

    def evaluate(self, m, H0):
        Mstar = M_Mobs(H0, self.Mstar_obs)
        return 0.4*log(10.0)*self.phistar \
               * pow(10.0, -0.4*(self.alpha+1.0)*(m-Mstar)) \
               * exp(-pow(10, -0.4*(m-Mstar)))

    def normalise(self, mmin, mmax):
        if self.norm is None:
            self.norm = quad(self.evaluate, mmin, mmax)[0]

    def pdf(self, m):
        return self.evaluate(m)/self.norm
        
    def __call__(self, m, H0):
        return self.evaluate(m,H0)


def M_Mobs(H0, M_obs):
    """
    Given an observed absolute magnitude, returns absolute magnitude
    """
    return M_obs + 5.*np.log10(H0/100.)
